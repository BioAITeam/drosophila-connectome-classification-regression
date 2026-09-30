# Computational resources and measured cost

Saved Iris hardware and timing records are in `5 Iris/5 Outputs/paper-results/revision-20260914T043414Z/resources/`: `hardware.json`, `execution_evidence.json`, `run_timings.csv` and `timing_summary.json`. These describe the original experiments, not minimum hardware requirements. Saved-model inference and paper reconstruction can run on CPU.

## Recorded hardware

| Resource | Capacity or execution setting |
| --- | --- |
| CPU | Two AMD EPYC 7302 processors; 32 physical cores and 64 threads in total |
| RAM | 503.74 GiB visible to the operating system |
| GPUs | Eight NVIDIA Quadro RTX 8000 devices, 48 GiB each |
| Iris CPU campaigns | 18 worker processes, normally one computational thread each |
| Full-graph campaign | One GPU per worker; 120 fits across three devices |

The host was shared. These capacities do not imply exclusive use of all hardware. The launcher admits a GPU when it has no active CUDA processes, at least 90% free memory and at most 10% utilization. It takes an advisory project lock and checks again before importing PyTorch; this does not reserve the device against unrelated applications.

The experiments used Python 3.12.13, PyTorch 2.2.2 with CUDA 12.1 and NumPy 1.26.4. Dependency versions and installation are in the [environment guide](<../1 Environment/README.md>).

## Iris training durations

Durations below are seconds per fit, reported as median [25th, 75th percentile]. Families combine the tasks, folds and seeds specified in the [Iris protocol](round4_protocol.md).

| Family | Device | Epoch budget | Timed fits | Duration, s |
| --- | --- | ---: | ---: | ---: |
| Subgraph rate models | CPU | 40 | 100 | 17.72 [17.12, 18.38] |
| Subgraph rate models and topology controls | CPU | 120 | 250 | 53.30 [52.25, 55.03] |
| Subgraph regression | CPU | 200 | 80 | 88.34 [86.78, 89.66] |
| Subgraph regression | CPU | Up to 600 | 80 | 256.57 [134.36, 275.24] |
| Full-graph classification | GPU | 40 | 40 | 307.80 [306.50, 312.33] |
| Full-graph classification and regression | GPU | 120 | 80 | 924.54 [918.75, 940.99] |
| Training-target shuffle | CPU | 40 | 25 | 17.70 [16.93, 18.10] |
| Supervised spiking variants | CPU | 40 | 250 | 32.43 [20.75, 38.92] |
| Local learning and its controls | CPU | 40 | 400 | 9.70 [7.95, 10.71] |
| Observed and globally permuted targets | CPU | 40 | 5,000 | 17.52 [17.22, 17.90] |
| Domain-transfer phases | CPU | 120 | 60 | 33.98 [29.20, 35.87] |

The 25 remaining runs were frozen-gain controls without parameter updates. They were not timed; their stored zeros are placeholders, not measured durations.

The timer covered optimization, epoch-level training/validation evaluation, diagnostics, history recording, saving final parameters and restoring selected gains. It excluded initial evaluation, data/graph preparation, model construction, test evaluation, bootstrap analysis, plotting and scheduling.

Measured intervals total **63.43 worker-hours**: 39.33 on CPUs and 24.10 on GPUs. Parallel intervals overlap, so this sum is not elapsed campaign time. CPU and GPU families differ in graph size and protocol and do not constitute a controlled hardware comparison.

## Optional full-graph profiling

To measure the model on another system, use a restored graph and a fresh output path:

```bash
./run.sh --device cuda --gpu 0 profile \
  --config "5 Iris/1 Configuration/tasks/iris.json" \
  --graph "2 Data/3 Processed/full.npz" \
  --output "5 Iris/6 Workspace/profiles/full-profile-new.json" --batches 2
```

Replace `0` with an available physical GPU index. The [profiler](<../3 Connectome/flybrain/profiling.py>) performs one warm-up update, two measured minibatch updates and a validation pass, without saving a trained checkpoint or evaluating test data. Its JSON separates graph loading, device transfer, forward/backward computation, gradient clipping, Adam and validation. CUDA timing waits for device completion.

The archived `5 Iris/5 Outputs/profiles/full-e001-b16-t24.json` used float32, batch size 16, 24 recurrent steps and 65,536-edge chunks on one Quadro RTX 8000. Its median measured minibatch took 1.078 s, validation took 0.279 s, and peak PyTorch allocated/reserved memory was 3.296/3.445 GiB. These are isolated profile measurements, not memory peaks or inference latency for the complete campaign. The companion `full-e001-b16-t24-chunk262144.json` records a different chunk size and device; it is not a controlled speedup benchmark.

The sparse implementation avoids a dense neuron-by-neuron matrix. With 25,582,938 edges, one float32 gain-parameter vector occupies 97.59 MiB; parameters, gradients and two Adam moment vectors together occupy about 390.36 MiB. Graph indices, recurrent activations, temporary buffers and allocator overhead require additional memory. Batch size, simulation duration and chunk size affect the total.
