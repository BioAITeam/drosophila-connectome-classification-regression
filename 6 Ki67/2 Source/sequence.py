"""Stream feature coordinates through existing visual neurons.

Each input neuron receives one coordinate per block, without a learned
projection. Backpropagation through time updates the inherited ``gain``
parameter. Gradient support is first order, as in :mod:`flybrain.model`.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from flybrain.model import SparseConnectome, _SparsePropagation


ProgressCallback = Callable[[dict[str, int]], None]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class SequenceConnectome(SparseConnectome):
    """Stream ``x[B,D]`` into the graph's existing annotated visual inputs.

    Coordinates are assigned to visual neurons in ascending unsigned body-ID
    order, in blocks of ``input_size`` (730 for the prepared MaleCNS subgraph).
    Each block is held for ``dwell_steps`` updates. State continues across all
    blocks and resets for every call. The endpoint population means of each
    block are weighted by its number of real coordinates, excluding padding.

    Non-reentrant checkpointing recomputes segments during backward without
    detaching their boundary states. ``checkpoint_blocks`` controls memory,
    not the simulation or training objective. Sparse matrices are built once
    per call and shared by every step and backward recomputation. The caller
    must finish backward before changing gains or graph buffers.

    ``progress_callback`` receives integer progress after each forward segment;
    it is not called during backward recomputation. It should return promptly.
    Float32 is the default dtype; ``.double()`` supports numerical verification.
    """

    def __init__(
        self,
        graph: dict[str, np.ndarray],
        *,
        dwell_steps: int = 4,
        checkpoint_blocks: int = 32,
        progress_callback: ProgressCallback | None = None,
        **model_kwargs: Any,
    ) -> None:
        dwell = _positive_integer(dwell_steps, "dwell_steps")
        segment_size = _positive_integer(checkpoint_blocks, "checkpoint_blocks")
        model_kwargs.setdefault("dynamics", "rate")
        model_kwargs.setdefault("dt", 0.5)
        model_kwargs.setdefault("leak", 1.0)
        model_kwargs.setdefault("input_amplitude", 2.0)
        model_kwargs.setdefault("temperature", 0.02)
        model_kwargs.setdefault("steps", 24)
        if model_kwargs["dynamics"] != "rate":
            raise ValueError("SequenceConnectome currently supports rate dynamics only")
        if model_kwargs.get("temporal_readout", False):
            raise ValueError("SequenceConnectome uses its fixed block-endpoint readout")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        super().__init__(graph, **model_kwargs)
        if np.any((np.asarray(graph["input_group"]) >= 0) & (np.asarray(graph["output_group"]) >= 0)):
            raise ValueError("Visual input and output populations must not overlap")
        inputs = np.flatnonzero(np.asarray(graph["input_group"]) >= 0)
        body_ids = np.asarray(graph["node_ids"], dtype=np.uint64)
        ordered = inputs[np.argsort(body_ids[inputs], kind="stable")]
        self.register_buffer("sequence_input_nodes", torch.from_numpy(ordered.astype(np.int64)))
        self.input_size = len(ordered)
        self.dwell_steps = dwell
        self.checkpoint_blocks = segment_size
        self.progress_callback = progress_callback

    def sequence_stats(self, feature_count: int) -> dict[str, int]:
        """Describe coverage and work without allocating a sequence or device."""
        dimensions = _positive_integer(feature_count, "feature_count")
        blocks = (dimensions + self.input_size - 1) // self.input_size
        return {
            "features": dimensions,
            "input_size": self.input_size,
            "blocks": blocks,
            "last_valid": dimensions - (blocks - 1) * self.input_size,
            "dwell_steps": self.dwell_steps,
            "recurrent_steps": blocks * self.dwell_steps,
            "segments": (blocks + self.checkpoint_blocks - 1) // self.checkpoint_blocks,
        }

    def _sequence_segment(
        self,
        state: Tensor,
        efficacy: Tensor,
        features: Tensor,
        *,
        matrix: Tensor,
        transpose: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute a segment without side effects for gradient checkpointing."""
        weighted_means = features.new_zeros((features.shape[0], 3))
        alpha = self.dt * self.leak
        # Bind invariant values once; raw images entail thousands of updates.
        input_nodes = self.sequence_input_nodes
        src, dst, base = self.plastic_src, self.plastic_dst, self.plastic_base
        edge_chunk = self.edge_chunk_size
        propagate = _SparsePropagation.apply
        for start in range(0, features.shape[1], self.input_size):
            block = features[:, start:start + self.input_size]
            valid = block.shape[1]
            current = features.new_zeros((features.shape[0], self.num_nodes))
            current = current.index_copy(1, input_nodes[:valid], block * self.input_amplitude)
            for _ in range(self.dwell_steps):
                transmitted = propagate(state, efficacy, matrix, transpose, src, dst, base, edge_chunk)
                drive = current + transmitted + self.tonic_current
                state = (1 - alpha) * state + alpha * torch.tanh(drive)
            weighted_means = weighted_means + self.population_means(state) * valid
        return state, weighted_means

    def population_readout(
        self,
        x: Tensor,
        checkpointed: bool | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Tensor:
        """Return the three fixed weighted population means, with a fresh state.

        ``x`` contains real coordinates only. The final incomplete block is
        implicitly zero-padded internally, and its weight uses its real length.
        Signed finite features are accepted; preprocessing belongs to the
        representation pipeline. No externally padded coordinates or lengths
        are inferred from feature values (a real zero remains a real feature).
        """
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("x must have nonempty shape [batch, features]")
        if x.dtype != self.gain.dtype or x.device != self.gain.device:
            raise ValueError("x must have the model's dtype and device")
        if not torch.isfinite(x).all():
            raise ValueError("input features must be finite")
        if checkpointed is not None and not isinstance(checkpointed, bool):
            raise ValueError("checkpointed must be bool or None")
        callback = self.progress_callback if progress_callback is None else progress_callback
        if callback is not None and not callable(callback):
            raise ValueError("progress_callback must be callable or None")
        use_checkpoint = checkpointed is not False and torch.is_grad_enabled() and (
            self.gain.requires_grad or x.requires_grad
        )
        efficacy = self.edge_efficacies()
        matrix, transpose = self._matrices(efficacy)
        segment = partial(self._sequence_segment, matrix=matrix, transpose=transpose)
        state = x.new_zeros((x.shape[0], self.num_nodes))
        weighted_means = x.new_zeros((x.shape[0], 3))
        features_per_segment = self.input_size * self.checkpoint_blocks
        total_blocks = (x.shape[1] + self.input_size - 1) // self.input_size
        for segment_index, start in enumerate(range(0, x.shape[1], features_per_segment), 1):
            stop = min(start + features_per_segment, x.shape[1])
            features = x[:, start:stop]
            if use_checkpoint:
                state, contribution = checkpoint(
                    segment, state, efficacy, features,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                state, contribution = segment(state, efficacy, features)
            weighted_means = weighted_means + contribution
            if callback is not None:
                completed_blocks = (stop + self.input_size - 1) // self.input_size
                callback({
                    "segment": segment_index,
                    "completed_blocks": completed_blocks,
                    "total_blocks": total_blocks,
                    "completed_features": stop,
                    "total_features": x.shape[1],
                    "recurrent_steps": completed_blocks * self.dwell_steps,
                })
        return weighted_means / x.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        return self.population_readout(x) / self.temperature

    def regression(self, x: Tensor) -> Tensor:
        means = self.population_readout(x)
        return means[:, 0] - means[:, 1]
