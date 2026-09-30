"""Fixed-interface recurrent models on measured connectome edges only.

The learned tensor ``gain`` stores one unconstrained parameter per selected
existing edge, transformed to a bounded efficacy by a sigmoid. The sparse
autograd kernel never constructs a dense N x N matrix or gathers the entire
batch x edge tensor. These equations are an
explicit computational hypothesis, not recovered neuronal physiology.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


class _SparsePropagation(torch.autograd.Function):
    """Exact first-order gradients of A(g) @ state, bounded edge workspace."""

    @staticmethod
    def forward(ctx: Any, state: Tensor, efficacy: Tensor, matrix: Tensor,
                transpose: Tensor, plastic_src: Tensor, plastic_dst: Tensor,
                plastic_base: Tensor, edge_chunk_size: int) -> Tensor:
        ctx.save_for_backward(state, matrix, transpose, plastic_src, plastic_dst,
                              plastic_base)
        ctx.edge_chunk_size = edge_chunk_size
        return torch.sparse.mm(matrix, state.transpose(0, 1).contiguous()).transpose(0, 1)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:
        state, matrix, transpose, src, dst, base = ctx.saved_tensors
        grad_state = grad_efficacy = None
        if ctx.needs_input_grad[0]:
            grad_state = torch.sparse.mm(
                transpose, grad_output.transpose(0, 1).contiguous()
            ).transpose(0, 1)
        if ctx.needs_input_grad[1]:
            grad_efficacy = torch.empty_like(base)
            for start in range(0, src.numel(), ctx.edge_chunk_size):
                end = min(start + ctx.edge_chunk_size, src.numel())
                # Workspaces have at most batch x edge_chunk_size entries.
                pre = state.index_select(1, src[start:end])
                post_gradient = grad_output.index_select(1, dst[start:end])
                grad_efficacy[start:end] = (
                    pre.mul(post_gradient).sum(dim=0) * base[start:end]
                )
        return grad_state, grad_efficacy, None, None, None, None, None, None


class _SurrogateSpike(torch.autograd.Function):
    """Hard threshold forward; triangular surrogate derivative backward."""

    @staticmethod
    def forward(ctx: Any, voltage_above_threshold: Tensor) -> Tensor:
        ctx.save_for_backward(voltage_above_threshold)
        return (voltage_above_threshold >= 0).to(voltage_above_threshold.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor]:
        (distance,) = ctx.saved_tensors
        return (grad_output * (1 - distance.abs()).clamp_min(0),)


def _array(graph: dict[str, np.ndarray], name: str, length: int,
           dtype: np.dtype) -> np.ndarray:
    if name not in graph:
        raise ValueError(f"graph is missing {name!r}")
    value = np.asarray(graph[name])
    if value.ndim != 1 or len(value) != length:
        raise ValueError(f"{name} must have shape ({length},), got {value.shape}")
    return np.array(value, dtype=dtype, copy=True)


class SparseConnectome(nn.Module):
    """A rate or LIF network with exactly the supplied nodes and edges.

    ``weight`` contains immutable anatomical contact counts. Functional base
    weights are these counts divided by total incoming counts, multiplied by
    the source neuron's supplied sign. ``gain`` controls selected efficacies
    in ``(0.05, 1.95)`` and initially gives efficacy 1.0. Other efficacies stay
    exactly 1.0. No bias, encoder, decoder, leak or temperature is learned.

    Inputs ``x[B,4]`` are already scaled intensities: this class does not fit
    preprocessing. Each annotated input neuron receives its group's scalar
    current; that is a four-patch visual proxy, not an anatomical retinal
    renderer. ``readout`` returns three fixed population means / temperature.
    Classification logits are converted to probabilities by the caller or loss.
    """

    efficacy_bounds = (0.05, 1.95)

    def __init__(self, graph: dict[str, np.ndarray], plastic_scope: str = "all",
                 dynamics: str = "rate", steps: int = 12, dt: float = 0.1,
                 leak: float = 1.0, input_amplitude: float = 2.0,
                 temperature: float = 0.1, edge_chunk_size: int = 65536,
                 threshold: float = 1.0, recurrent_scale: float = 1.0,
                 tonic_current: float = 0.0, temporal_readout: bool = False) -> None:
        super().__init__()
        if plastic_scope not in {"all", "kc_mbon"}:
            raise ValueError("plastic_scope must be 'all' or 'kc_mbon'")
        if dynamics not in {"rate", "lif"}:
            raise ValueError("dynamics must be 'rate' or 'lif'")
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")
        if int(edge_chunk_size) != edge_chunk_size or edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be a positive integer")
        for name, value in (("dt", dt), ("leak", leak), ("temperature", temperature),
                            ("threshold", threshold), ("recurrent_scale", recurrent_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(input_amplitude) or input_amplitude < 0:
            raise ValueError("input_amplitude must be finite and non-negative")
        if not math.isfinite(tonic_current) or tonic_current < 0:
            raise ValueError("tonic_current must be finite and non-negative")
        if dt * leak > 1:
            raise ValueError("dt * leak must be <= 1 for the explicit update")
        if "node_ids" not in graph or np.asarray(graph["node_ids"]).ndim != 1:
            raise ValueError("node_ids must be a one-dimensional array")
        ids = np.array(graph["node_ids"], dtype=np.uint64, copy=True)
        n = len(ids)
        if n == 0 or len(np.unique(ids)) != n:
            raise ValueError("node_ids must be nonempty and unique")
        if "src" not in graph or np.asarray(graph["src"]).ndim != 1:
            raise ValueError("src must be a one-dimensional array")
        e = len(graph["src"])
        src = _array(graph, "src", e, np.int64)
        dst = _array(graph, "dst", e, np.int64)
        counts = _array(graph, "weight", e, np.float32)
        sign = _array(graph, "sign", n, np.float32)
        input_group = _array(graph, "input_group", n, np.int64)
        output_group = _array(graph, "output_group", n, np.int64)
        reward = _array(graph, "reward_mask", n, np.bool_)
        punishment = _array(graph, "punishment_mask", n, np.bool_)
        plastic_mask = _array(graph, "plastic_mask", e, np.bool_)
        if e == 0 or np.any(src < 0) or np.any(dst < 0) or np.any(src >= n) or np.any(dst >= n):
            raise ValueError("edges must be nonempty and refer to existing node indices")
        if not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError("anatomical weights must be finite positive counts")
        if not np.isin(sign, (-1.0, 0.0, 1.0)).all():
            raise ValueError("sign must contain only -1, 0 or +1")
        for name, groups, count in (("input_group", input_group, 4),
                                     ("output_group", output_group, 3)):
            if np.any(groups < -1) or np.any(groups >= count):
                raise ValueError(f"{name} must be -1 or a valid group index")
            if any(not np.any(groups == k) for k in range(count)):
                raise ValueError(f"{name} must contain every one of its {count} groups")
        if np.any(reward & punishment):
            raise ValueError("reward and punishment populations must be disjoint")

        self.num_nodes = n
        self.num_edges = e
        self.plastic_scope = plastic_scope
        self.dynamics = dynamics
        self.steps = int(steps)
        self.dt = float(dt)
        self.leak = float(leak)
        self.input_amplitude = float(input_amplitude)
        self.temperature = float(temperature)
        self.threshold = float(threshold)
        self.recurrent_scale = float(recurrent_scale)
        self.edge_chunk_size = int(edge_chunk_size)
        self.tonic_current = float(tonic_current)
        self.temporal_readout = bool(temporal_readout)

        # torch 2.2 cannot store uint64. Int64 bit views retain every ID bit;
        # node_ids_numpy() restores the original uint64 without conversion.
        self.register_buffer("node_ids", torch.from_numpy(ids.view(np.int64)))
        for name, value in (("src", src), ("dst", dst), ("weight", counts),
                            ("sign", sign), ("input_group", input_group),
                            ("output_group", output_group), ("reward_mask", reward),
                            ("punishment_mask", punishment), ("plastic_mask", plastic_mask)):
            self.register_buffer(name, torch.from_numpy(value))

        incoming = np.bincount(dst, weights=counts, minlength=n)
        base = (counts / incoming[dst] * sign[src] * recurrent_scale).astype(np.float32)
        self.register_buffer("base_weight", torch.from_numpy(base))
        selected = np.arange(e, dtype=np.int64) if plastic_scope == "all" else np.flatnonzero(plastic_mask)
        if len(selected) == 0:
            raise ValueError("plastic_scope selects no existing edges")
        self.register_buffer("plastic_edge_indices", torch.from_numpy(selected))
        self.register_buffer("plastic_src", self.src[self.plastic_edge_indices].clone())
        self.register_buffer("plastic_dst", self.dst[self.plastic_edge_indices].clone())
        self.register_buffer("plastic_base", self.base_weight[self.plastic_edge_indices].clone())
        self.gain = nn.Parameter(torch.zeros(len(selected), dtype=torch.float32))

        # CSR rows are postsynaptic cells; the inverse layout is prepared once.
        # Sparse mm accepts unsorted columns, and preserves duplicate entries.
        order = np.argsort(dst, kind="stable").astype(np.int64)
        reverse = np.argsort(src, kind="stable").astype(np.int64)
        row_ptr = np.r_[0, np.cumsum(np.bincount(dst, minlength=n))].astype(np.int64)
        reverse_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=n))].astype(np.int64)
        self.register_buffer("csr_order", torch.from_numpy(order))
        self.register_buffer("csr_ptr", torch.from_numpy(row_ptr))
        self.register_buffer("csr_src", torch.from_numpy(src[order].copy()))
        self.register_buffer("transpose_order", torch.from_numpy(reverse))
        self.register_buffer("transpose_ptr", torch.from_numpy(reverse_ptr))
        self.register_buffer("transpose_dst", torch.from_numpy(dst[reverse].copy()))
        self.register_buffer("input_nodes", torch.from_numpy(np.flatnonzero(input_group >= 0)))
        self.register_buffer("input_features", self.input_group[self.input_nodes].clone())
        for group in range(3):
            self.register_buffer(f"output_nodes_{group}", torch.from_numpy(np.flatnonzero(output_group == group)))

    @property
    def num_plastic_edges(self) -> int:
        return self.gain.numel()

    def node_ids_numpy(self) -> np.ndarray:
        return self.node_ids.detach().cpu().numpy().view(np.uint64).copy()

    def edge_efficacies(self) -> Tensor:
        """Efficacies for selected edges, in plastic_edge_indices order."""
        low, high = self.efficacy_bounds
        return low + (high - low) * self.gain.sigmoid()

    def effective_edge_weights(self) -> Tensor:
        """Functional weights for all original edges, preserving their order."""
        result = self.base_weight.clone()
        return result.index_copy(0, self.plastic_edge_indices,
                                 self.plastic_base * self.edge_efficacies())

    @torch.no_grad()
    def apply_efficacy_delta(self, delta: Tensor) -> None:
        """Apply a local-rule update, clipping inside bounds; no topology edit."""
        delta = torch.as_tensor(delta, dtype=self.gain.dtype, device=self.gain.device)
        if delta.shape != self.gain.shape or not torch.isfinite(delta).all():
            raise ValueError("delta must be finite and have one value per plastic edge")
        low, high = self.efficacy_bounds
        probability = (self.edge_efficacies() + delta - low) / (high - low)
        probability = probability.clamp(1e-6, 1 - 1e-6)
        self.gain.copy_(torch.logit(probability))

    def fixed_current(self, x: Tensor) -> Tensor:
        """Encode four already scaled features with an immutable group map."""
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError("x must have shape [batch, 4]")
        if x.device != self.gain.device or x.dtype != self.gain.dtype:
            raise ValueError("x must have the model's dtype and device")
        if not torch.isfinite(x).all():
            raise ValueError("input features must be finite")
        current = x.new_zeros((len(x), self.num_nodes))
        return current.index_copy(1, self.input_nodes,
                                  x[:, self.input_features] * self.input_amplitude)

    def _matrices(self, efficacy: Tensor) -> tuple[Tensor, Tensor]:
        # Autograd is provided by _SparsePropagation, which receives efficacy
        # separately. Both sparse layouts are shared across the entire rollout.
        with torch.no_grad():
            values = self.base_weight.clone()
            values.index_copy_(0, self.plastic_edge_indices, self.plastic_base * efficacy)
            matrix = torch.sparse_csr_tensor(
                self.csr_ptr, self.csr_src, values[self.csr_order],
                size=(self.num_nodes, self.num_nodes), device=values.device)
            transpose = torch.sparse_csr_tensor(
                self.transpose_ptr, self.transpose_dst, values[self.transpose_order],
                size=(self.num_nodes, self.num_nodes), device=values.device)
        return matrix, transpose

    def simulate(self, x: Tensor, initial_state: Tensor | tuple[Tensor, Tensor] | None = None,
                 extra_current: Tensor | None = None, steps: int | None = None,
                 return_trajectory: bool = False,
                 return_internal_state: bool = False) -> Any:
        """Return activity [B,N], or all activities [T,B,N].

        Every call starts at zero unless ``initial_state`` is explicit. For a
        rate model it is [B,N]; for LIF it must be (activity, membrane). Set
        ``return_internal_state`` to return ``(result, continuation_state)``.
        LIF spike activity alone cannot continue its membrane state faithfully.

        ``extra_current`` may have shape [B,N] (constant) or [T,B,N] (pulses).
        It enters the actual neuronal dynamics. No reward arrives implicitly.
        """
        count = self.steps if steps is None else steps
        if int(count) != count or count < 1:
            raise ValueError("steps must be a positive integer")
        count = int(count)
        current = self.fixed_current(x)
        if extra_current is not None:
            valid_shapes = {tuple(current.shape), (count, *current.shape)}
            if tuple(extra_current.shape) not in valid_shapes:
                raise ValueError("extra_current must have shape [B,N] or [steps,B,N]")
            if extra_current.dtype != x.dtype or extra_current.device != x.device:
                raise ValueError("extra_current must have the model's dtype and device")
            if not torch.isfinite(extra_current).all():
                raise ValueError("extra_current must be finite")
        state = current.new_zeros(current.shape)
        membrane = current.new_zeros(current.shape)
        if initial_state is not None:
            if self.dynamics == "lif":
                if not isinstance(initial_state, tuple) or len(initial_state) != 2:
                    raise ValueError("LIF initial_state must be (activity, membrane)")
                state, membrane = initial_state
            else:
                if not isinstance(initial_state, Tensor):
                    raise ValueError("rate initial_state must be a tensor")
                state = initial_state
            for candidate in (state, membrane):
                if candidate.shape != current.shape or candidate.device != x.device or candidate.dtype != x.dtype:
                    raise ValueError("initial state must match the model's [B,N], device and dtype")
                if not torch.isfinite(candidate).all():
                    raise ValueError("initial state must be finite")
        efficacy = self.edge_efficacies()
        matrix, transpose = self._matrices(efficacy)
        trajectory = []
        for step in range(count):
            transmitted = _SparsePropagation.apply(
                state, efficacy, matrix, transpose, self.plastic_src,
                self.plastic_dst, self.plastic_base, self.edge_chunk_size)
            drive = current + transmitted + self.tonic_current
            if extra_current is not None:
                drive = drive + (extra_current if extra_current.ndim == 2 else extra_current[step])
            if self.dynamics == "rate":
                alpha = self.dt * self.leak
                state = (1 - alpha) * state + alpha * torch.tanh(drive)
            else:
                membrane = membrane + self.dt * (drive - self.leak * membrane)
                state = _SurrogateSpike.apply(membrane - self.threshold)
                membrane = membrane * (1 - state.detach())
            if return_trajectory:
                trajectory.append(state)
        result = torch.stack(trajectory) if return_trajectory else state
        if return_internal_state:
            continuation = (state, membrane) if self.dynamics == "lif" else state
            return result, continuation
        return result

    def rollout(self, x: Tensor, **kwargs: Any) -> Any:
        """Alias of simulate for explicit stimulation protocols."""
        return self.simulate(x, **kwargs)

    def population_means(self, activity: Tensor) -> Tensor:
        """Measure existing populations; a trajectory is averaged over time."""
        if activity.ndim == 3:
            activity = activity.mean(dim=0)
        if activity.ndim != 2 or activity.shape[1] != self.num_nodes:
            raise ValueError("activity must be [B,N] or [T,B,N]")
        return torch.stack([
            activity.index_select(1, getattr(self, f"output_nodes_{group}")).mean(dim=1)
            for group in range(3)
        ], dim=1)

    def readout(self, activity: Tensor) -> Tensor:
        return self.population_means(activity) / self.temperature

    def forward(self, x: Tensor, return_activity: bool = False) -> Any:
        activity = self.simulate(x, return_trajectory=self.temporal_readout)
        logits = self.readout(activity)
        return (logits, activity) if return_activity else logits

    def regression(self, x: Tensor) -> Tensor:
        means = self.population_means(self.simulate(x, return_trajectory=self.temporal_readout))
        return means[:, 0] - means[:, 1]
