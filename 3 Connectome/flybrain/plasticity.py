"""Local node-perturbation plasticity with anatomically restricted feedback.

This is a modeling hypothesis, not measured fly STDP or a validated
dopamine model. No backpropagation, learned critic, encoder or decoder is used.
"""
from __future__ import annotations

import math

import torch


class LocalRewardLearner:
    """Three-factor efficacy updates on existing plastic edges only.

    The three factors are recent presynaptic activity, an exploratory current
    perturbation in the existing postsynaptic cell, and evoked DAN modulation.
    Modulation reaches a post cell only through measured DAN -> post edges.
    Its valence (+ reward DAN, - punishment DAN) is a fixed modeling choice.

    ``step`` is the training protocol. ``predict`` is deterministic test with
    no label, exploration or feedback. ``step(update=False)`` also runs this
    test path; use ``learning_rate=0`` for a frozen training control that keeps
    matched exploratory currents and reward pulses.
    """

    def __init__(self, model, *, learning_rate=.01, exploration=.15,
                 eligibility_decay=.8, pulse_amplitude=1., pulse_steps=3,
                 chunk_size=32768):
        for name, value in (("learning_rate", learning_rate),
                            ("pulse_amplitude", pulse_amplitude)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(exploration) or exploration <= 0:
            raise ValueError("exploration must be finite and > 0")
        if not math.isfinite(eligibility_decay) or not 0 <= eligibility_decay < 1:
            raise ValueError("eligibility_decay must be in [0,1)")
        for name, value in (("pulse_steps", pulse_steps), ("chunk_size", chunk_size)):
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if getattr(model, "dynamics", "rate") != "rate":
            raise ValueError("Local node-perturbation requires rate dynamics; LIF is not implemented here")
        self.model = model
        self.learning_rate = float(learning_rate)
        self.exploration = float(exploration)
        self.decay = float(eligibility_decay)
        self.pulse_amplitude = float(pulse_amplitude)
        self.pulse_steps = int(pulse_steps)
        self.chunk_size = int(chunk_size)
        # Read interfaces from immutable model buffers, not mutable graph arrays.
        self.src = model.src
        self.dst = model.dst
        self.plastic_ids = model.plastic_edge_indices
        self.plastic_src = model.plastic_src
        self.plastic_dst = model.plastic_dst
        self.plastic_base = model.plastic_base
        self.reward_cells = model.reward_mask
        self.punishment_cells = model.punishment_mask
        self.n = model.num_nodes
        if not self.reward_cells.any() or not self.punishment_cells.any():
            raise ValueError("Local feedback requires existing reward and punishment DAN populations")
        self.perturb_cells = torch.unique(self.plastic_dst)
        counts = model.weight
        self.dan_edge_ids = torch.nonzero(
            (self.reward_cells | self.punishment_cells)[self.src], as_tuple=False
        ).flatten()
        self.dan_src = self.src[self.dan_edge_ids]
        self.dan_dst = self.dst[self.dan_edge_ids]
        denominator = counts.new_zeros(self.n)
        denominator.index_add_(0, self.dan_dst, counts[self.dan_edge_ids])
        self.dan_weight = counts[self.dan_edge_ids] / denominator[self.dan_dst].clamp_min(1.)
        reward_reach = counts.new_zeros(self.n)
        punishment_reach = counts.new_zeros(self.n)
        from_reward = self.reward_cells[self.dan_src]
        reward_reach.index_add_(0, self.dan_dst[from_reward], counts[self.dan_edge_ids[from_reward]])
        punishment_reach.index_add_(0, self.dan_dst[~from_reward], counts[self.dan_edge_ids[~from_reward]])
        reachable = denominator[self.plastic_dst] > 0
        reward_reachable = reward_reach[self.plastic_dst] > 0
        punishment_reachable = punishment_reach[self.plastic_dst] > 0
        if not reachable.any():
            raise ValueError("No measured DAN→plastic postsynaptic path; cannot invent a feedback connection")
        if not reward_reachable.any() or not punishment_reachable.any():
            raise ValueError("Both DAN valences need measured edges to at least one plastic postsynaptic cell")
        self.coverage = float(reachable.float().mean())
        self.reward_coverage = float(reward_reachable.float().mean())
        self.punishment_coverage = float(punishment_reachable.float().mean())
        self.joint_coverage = float((reward_reachable & punishment_reachable).float().mean())

    @staticmethod
    def _validate_task(task, n_classes):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if task == "classification" and (int(n_classes) != n_classes or not 2 <= n_classes <= 3):
            raise ValueError("classification requires two or three fixed output groups")

    @torch.no_grad()
    def predict(self, x, *, task="classification", n_classes=3):
        """No labels, random currents, feedback pulses or weight updates."""
        self._validate_task(task, n_classes)
        state = self.model.simulate(x, return_trajectory=self.model.temporal_readout)
        if task == "classification":
            return self.model.readout(state)[:, :n_classes].argmax(dim=1)
        return self._regression_readout(state)

    def _stats(self, reward, predictions, *, changed=0, update_l2=0.,
               dan_evoked_mean=0., reward_evoked_mean=0., punishment_evoked_mean=0.,
               eligibility_l2=0., modulation_mean_absolute=0., positive_update_fraction=0.):
        return {
            "reward": reward.cpu().tolist(), "predictions": predictions.cpu().tolist(),
            "changed_edges": changed, "update_l2": update_l2,
            "eligibility_l2": eligibility_l2,
            "modulation_mean_absolute": modulation_mean_absolute,
            "positive_update_fraction": positive_update_fraction,
            "dan_evoked_mean": dan_evoked_mean,
            "reward_dan_evoked_mean": reward_evoked_mean,
            "punishment_dan_evoked_mean": punishment_evoked_mean,
            "dan_to_plastic_coverage": self.coverage,
            "reward_dan_to_plastic_coverage": self.reward_coverage,
            "punishment_dan_to_plastic_coverage": self.punishment_coverage,
            "both_valences_to_plastic_coverage": self.joint_coverage,
        }

    @torch.no_grad()
    def step(self, x, target, *, task="classification", n_classes=3,
             reward_override=None, update=True, generator=None):
        self._validate_task(task, n_classes)
        m = self.model
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must contain a nonempty batch")
        batch = len(x)
        target = torch.as_tensor(target, device=x.device)
        if target.shape != (batch,) or not torch.isfinite(target).all():
            raise ValueError("target must be finite with one value per presented sample")
        if task == "classification" and (
            torch.any(target != target.round()) or torch.any(target < 0) or torch.any(target >= n_classes)
        ):
            raise ValueError("classification targets must index the declared fixed output groups")
        override = None
        if reward_override is not None:
            override = torch.as_tensor(reward_override, device=x.device, dtype=x.dtype)
            if override.shape != (batch,) or not torch.isfinite(override).all():
                raise ValueError("reward_override must be finite with one value per presented sample")
        if not update:
            predictions = self.predict(x, task=task, n_classes=n_classes)
            if task == "classification":
                reward = torch.where(predictions == target, 1., -1.).to(dtype=x.dtype)
            else:
                # The deterministic response is its own reference; no trial's
                # exploratory improvement is evaluated on the test path.
                reward = x.new_zeros(batch)
            if override is not None:
                reward = override
            return self._stats(reward, predictions)

        noise = x.new_zeros((batch, self.n))
        noise[:, self.perturb_cells] = torch.randn(
            (batch, len(self.perturb_cells)), device=x.device, dtype=x.dtype,
            generator=generator
        ) * self.exploration
        trajectory = m.simulate(x, extra_current=noise, return_trajectory=True)
        state = trajectory[-1]
        readout_activity = trajectory if m.temporal_readout else state
        if task == "classification":
            predictions = m.readout(readout_activity)[:, :n_classes].argmax(dim=1)
            reward = torch.where(predictions == target, 1., -1.).to(dtype=x.dtype)
        else:
            predictions = self._regression_readout(readout_activity)
            # Two responses of this same connectome: no trained baseline/critic.
            baseline = self._regression_readout(m.simulate(x, return_trajectory=m.temporal_readout))
            reward = ((baseline - target).square() - (predictions - target).square()).clamp(-1., 1.)
        if override is not None:
            reward = override
        # Exact no-reinforcement control, including no sigmoid/logit round trip.
        if not torch.any(reward != 0) or self.pulse_amplitude == 0:
            return self._stats(reward, predictions)
        pulse = (
            reward.clamp_min(0)[:, None] * self.reward_cells[None, :]
            + (-reward).clamp_min(0)[:, None] * self.punishment_cells[None, :]
        ) * self.pulse_amplitude
        # Paired responses start at the same post-stimulus fast state. Only
        # the explicit DAN current differs; endogenous activity is subtracted.
        zero_x = torch.zeros_like(x)
        feedback = m.simulate(zero_x, initial_state=state, extra_current=pulse,
                              steps=self.pulse_steps)
        no_feedback = m.simulate(zero_x, initial_state=state, steps=self.pulse_steps)
        evoked = (feedback - no_feedback).clamp_min(0)
        modulation = torch.zeros_like(state)
        for start in range(0, self.dan_src.numel(), self.chunk_size):
            end = min(start + self.chunk_size, self.dan_src.numel())
            dan_src = self.dan_src[start:end]
            valence = torch.where(self.reward_cells[dan_src], 1., -1.).to(dtype=x.dtype)
            modulation.index_add_(
                1, self.dan_dst[start:end],
                evoked[:, dan_src] * self.dan_weight[None, start:end] * valence[None, :]
            )
        deltas = torch.zeros_like(m.edge_efficacies())
        eligibility_squared = 0.
        if self.learning_rate >= 0:
            t_weights = self.decay ** torch.arange(
                len(trajectory) - 1, -1, -1, device=x.device, dtype=x.dtype
            )
            t_weights = t_weights / t_weights.sum()
            for start in range(0, len(self.plastic_src), self.chunk_size):
                end = min(start + self.chunk_size, len(self.plastic_src))
                pre = self.plastic_src[start:end]
                post = self.plastic_dst[start:end]
                eligibility = x.new_zeros((batch, end - start))
                for t in range(1, len(trajectory)):
                    # The presynaptic response precedes the perturbed post by
        # one integration step. Indexing preserves the [B,E] dimensions.
                    eligibility += (
                        t_weights[t] * trajectory[t - 1].index_select(1, pre)
                        * noise.index_select(1, post) / self.exploration**2
                    )
                # Efficacy multiplies an existing signed functional base weight;
                # the base factor is essential for inhibitory edge orientation.
                eligibility_squared += float(eligibility.square().sum())
                delta = self.learning_rate * self.plastic_base[start:end] * (
                    eligibility * modulation.index_select(1, post)
                ).mean(0)
                deltas[start:end] = delta.clamp(-.05, .05)
        changed = 0
        update_l2 = 0.
        if torch.any(deltas != 0):
            before = m.edge_efficacies().clone()
            inactive = deltas == 0
            inactive_gain = m.gain[inactive].clone()
            m.apply_efficacy_delta(deltas)
            # Exact invariance for non-modulated connections despite inverse
            # sigmoid rounding in a vectorized efficacy update.
            m.gain[inactive] = inactive_gain
            actual = m.edge_efficacies() - before
            changed = int(torch.count_nonzero(actual))
            update_l2 = float(torch.linalg.vector_norm(actual))
        dan_cells = self.reward_cells | self.punishment_cells
        return self._stats(
            reward, predictions, changed=changed, update_l2=update_l2,
            dan_evoked_mean=float(evoked[:, dan_cells].mean()),
            reward_evoked_mean=float(evoked[:, self.reward_cells].mean()),
            punishment_evoked_mean=float(evoked[:, self.punishment_cells].mean()),
            eligibility_l2=math.sqrt(eligibility_squared),
            modulation_mean_absolute=float(modulation[:, self.perturb_cells].abs().mean()),
            positive_update_fraction=float((deltas > 0).float().mean()),
        )

    def _regression_readout(self, state):
        means = self.model.population_means(state)
        return means[:, 0] - means[:, 1]
