# Connectome model and fixed interfaces

[`SparseConnectome`](<../3 Connectome/flybrain/model.py>) represents the nodes and directed edges in a prepared graph. This document describes its Iris interface and shared recurrent dynamics. Ki-67 extends it with [`SequenceConnectome`](<../6 Ki67/2 Source/sequence.py>), which supplies feature coordinates in blocks and combines block-endpoint readouts; see the [Ki-67 protocol](<../6 Ki67/4 Documentation/protocol.md>).

## What is learned

The only `nn.Parameter` is a tensor named `gain`, with one element for each selected existing edge. `plastic_scope="all"` selects every retained edge; `plastic_scope="kc_mbon"` selects the anatomical KC-to-MBON mask. An empty selected mask is rejected. The model adds no neurons, edges, trainable biases, input projections, output projections, temperature, or leak parameters.

The stored `gain` tensor is the **unconstrained parameter theta**, not the effective gain itself:

```text
g_e = 0.05 + 1.90 * sigmoid(theta_e)
base_e = recurrent_scale * sign[src_e] * count_e / sum(count_incoming_to_dst_e)
w_e = base_e * g_e        for selected plastic edges
w_e = base_e              for other edges
```

Theta starts at zero, so the initial effective gain is 1.0. Gains stay positive and preserve the specified sign. Original anatomical contact counts remain unchanged. Normalization, signs, and neuronal equations are simulation assumptions, not measured physiology. A zero presynaptic sign disables the modeled fast transmission on that edge; retaining a dopaminergic neuron does not make its fast output excitatory.

`apply_efficacy_delta(delta)` receives changes to the effective gains in `plastic_edge_indices` order, clips the permitted range, and updates only the corresponding stored parameters. The [local learning rule](plasticity.md) uses activity in annotated modulatory cells through existing contacts.

## Input current and output readout

`fixed_current(x)` receives already scaled intensities with shape `[B, 4]`. An input neuron receives `input_amplitude * x[:, input_group]`; other neurons receive no direct dataset current. The four channels are fixed bilateral R7/R8 population assignments, or the declared direct-KC diagnostic interface. This is a numerical stimulation proxy, without a reconstructed retinotopic camera. Scaling is fitted only on training observations outside the neural model.

`population_means(activity)` averages each output population: G0 contains four MBON09 neurons, G1 contains two MBON27 neurons and G2 contains two MBON05 neurons. `readout(activity)` divides these means by a fixed temperature to produce logits. Softmax introduces no trainable parameters or neurons. Output membership and its class correspondence are fixed before training. `regression(x)` returns `mean(G0) - mean(G1)`; conversion back to physical units uses the fixed training-derived target scale, without a learned regression head.

`forward` uses the final recurrent step by default. A trajectory passed to `readout`, with shape `[T, B, N]`, is averaged over time. A comparison that changes the readout window must declare that change.

## Neuronal dynamics

The rate model uses a signed, normalized continuous activity state and an Euler update:

```text
drive = fixed_current(x) + W @ activity + tonic_current + extra_current(t)
alpha = dt * leak
activity_next = (1 - alpha) * activity + alpha * tanh(drive)
```

The fixed `tonic_current` and optional extra current are zero in the supervised rate reference; complementary local and spiking protocols specify their own currents. Both `dt` and `leak` are fixed, with `dt * leak <= 1`. Negative activity is permitted; these states are not firing rates in hertz. `recurrent_scale` is a fixed simulation setting.

The leaky integrate-and-fire (LIF) variant uses:

```text
membrane_next = membrane + dt * (drive - leak * membrane)
spike = 1[membrane_next >= threshold]
membrane_next *= (1 - spike)
```

The forward pass produces binary spikes. Supervised training substitutes the triangular derivative `max(0, 1 - abs(membrane - threshold))` for the threshold derivative and detaches the spike used in membrane reset. This is an approximate gradient. The implementation does not include delays, refractory periods, conductances, morphology, or cell-type-specific time constants.

## Simulation API and state

The following calls use the reference Iris rate configuration with a loaded graph and input tensor `x`; `B`, `T` and `N` denote batch size, simulation steps and neuron count.

```python
from flybrain.model import SparseConnectome

model = SparseConnectome(
    graph, plastic_scope="all", dynamics="rate", steps=24,
    dt=0.5, leak=1.0, input_amplitude=2.0, temperature=0.02,
)
logits = model(x)                                      # [B, 3]
logits, activity = model(x, return_activity=True)
trajectory = model.simulate(x, return_trajectory=True)  # [T, B, N]
logits_from_window = model.readout(trajectory)
prediction = model.regression(x)                       # [B]

activity, continuation = model.simulate(x, return_internal_state=True)
next_activity = model.simulate(
    x, initial_state=continuation, extra_current=neuron_pulses
)
```

`rollout` aliases `simulate`. Additional current has shape `[B, N]` or `[T, B, N]`, on the same device and with the same dtype, and enters the neuronal equations directly.

Each call starts with zero transient state unless `initial_state` is passed explicitly. Rate continuation uses an activity tensor `[B, N]`. LIF continuation requires both `(activity, membrane)` and rejects spikes alone, which would lose the membrane state. The model retains no mutable transient state between calls.

Teacher pulses alone do not change parameters. An explicit learning rule must calculate an update and call `apply_efficacy_delta`; teacher pulses and updates are disabled at test. The supervised optimizer also changes only `gain`. In either case, persistent adaptation is stored in internal connection efficacies. **Activity reset** clears transient neuronal state while retaining gains. It is different from **Target only** transfer training, whose saved key is `reset`.

## Sparse computation and checkpoints

The sparse operator uses compressed sparse row (CSR) representations for transmission and state gradients. Effective values are prepared once per rollout and reused across steps. For each propagation step, the derivative with respect to an effective gain sums `activity_pre * gradient_post * base_weight` across the batch. Autograd applies the sigmoid derivative to obtain the gradient of its stored theta parameter. Selected edges are processed in chunks, with default `edge_chunk_size=65536`; no dense `N x N` connectivity matrix or full batch-by-edge tensor is constructed. Validated gradient support is first order.

CSR indices, both orientations, provenance buffers, recurrent activations, and optimizer moments all consume memory. Explicit trajectories require `T * B * N` values. Sparse storage alone does not guarantee that an arbitrary batch and duration fit a device. See [computational cost](full_graph_compute.md) for measurements.

The model's `state_dict` includes anatomical buffers and the parameter tensor; constructor settings are retained in the experiment configuration. Original training exports package selected and initial parameters in `model.pt` with configuration and identity checks. The publication archive uses lossless shared storage for repeated initialization and edge-index tensors, located through each checkpoint's `model.storage.json`. Restore the shared tensors, graph and dataset metadata with the selected checkpoint, as specified in the [artifact guide](<../4 Trained Models/README.md>).

For compatibility with PyTorch 2.2 buffers, original `uint64` node identifiers are represented by an `int64` view with identical bits. `node_ids_numpy()` reconstructs the original `uint64` IDs, including values above `2**63 - 1`. Original node and edge ordering is preserved; CSR orderings are additional execution views.
