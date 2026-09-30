# Iris paper protocol

The paper campaign contains **6,390 completed computational runs**, specified by the seven frozen plans in `5 Iris/1 Configuration/`. The `round4` prefix identifies this campaign in the source and artifacts. Earlier Iris exploration informed its design, so this is an internally controlled evaluation, not untouched external validation. The archived run inventory records every completed fit and its experimental condition.

## Tasks and fixed interfaces

All runs use the same 150 Iris observations. Species classification has three classes and all four measurements as inputs. Four independent regressions each predict one measurement in centimeters from the other three, preserving their original order and padding the fourth input channel with zero. Species labels are never regression inputs.

The [model](model.md) defines four fixed input channels, three MBON output populations and bounded gains on existing connections. Classification uses the three population means; regression uses the first-minus-second difference, converted back to centimeters. External input and output mappings remain fixed.

The [evaluation protocol](evaluation.md) defines grouped splitting and training-derived scales. Each complete out-of-fold (OOF) evaluation combines five fitted models to predict each of the 150 flowers once from a held-out fold.

## Experimental matrix

The [plan index](benchmark_matrix.md) lists all seven plan files and their run counts. The anatomical and full-graph campaigns contain 510 and 120 runs; global permutations contribute 5,000. The remaining 760 runs cover frozen/training-target controls, spiking dynamics, local plasticity and transfer.

## Anatomical graph, budgets, and topology

The anatomical subgraph contains 1,323 neurons and 40,755 edges. Every task has 40- and 120-epoch budgets, crossing partition seeds 1729/2718, optimization seeds 0/1, and five outer folds: 20 fits per task and budget, yielding four complete OOF evaluations. The four regressions also use budgets of 200 and up to 600 epochs. These contribute 200 basic anatomical fits and 160 extended regression fits.

Degree-only and restricted rewiring each use graph seeds 137/241/997, a 120-epoch budget, partition seed 1729, optimization seed zero, and five folds for every task, contributing 150 fits. Graphs are prepared before fitting, using only anatomical data. Constraints, accepted swaps, and realized changed-destination fractions are recorded in each graph's sidecar. These are partial artificial perturbations, not uniformly random samples of all possible graphs. The [graph-control documentation](graph_controls.md) distinguishes the properties preserved by each rewiring rule.

## Full graph and input matching

Both full-graph interfaces retain 166,700 neurons and 25,582,938 edges. Native inputs assign all 1,783 eligible R7/R8 IDs to the four channels. Matched inputs assign the exact 730 IDs and channels of the subgraph without removing any full-graph nodes or edges. All eight output MBON IDs and their population assignments are retained.

For regression, the fourth channel is zero. The three data channels therefore map to 1,333 native or 560 matched/subgraph input IDs. Individual scaled feature values can also be zero. These counts describe interface assignments, not the total number of neurons activated through recurrence.

Full-graph classification crosses both interfaces, 40/120 epochs, and the same 20 fits per condition, for 80 runs. Each full-graph regression uses 120 epochs, both interfaces, and one five-fold evaluation at partition/optimization seeds 1729/0, for 40 more runs.

The common 120-epoch comparison uses subgraph/730 inputs, full graph/730 inputs, and full graph/1,783 inputs for every task. Classification includes all four shared OOF evaluations. Each regression uses only the one evaluation shared by all three graph conditions. Additional subgraph repetitions are excluded from that matched regression comparison. Means and sample standard deviations are descriptive; one OOF evaluation cannot estimate variation between evaluations.

`5 Iris/5 Outputs/campaigns/round4-comparison/` aggregates the 630 core and full runs without fitting new models. The common-120 subset uses 120 of those fits and 24 OOF evaluations. Comparisons require matched partition/optimization seeds and actual test assignments. Matching full-graph stimulation controls the input assignment within the full graph. Its comparison with the subgraph still changes surrounding connections and incoming-contact normalization, so it does not isolate neuron count alone.

## Optimization and checkpoint selection

The supervised rate reference uses 24 recurrent steps, `dt=0.5`, leak 1, input amplitude 2, fixed temperature 0.02, batch size 16, Adam learning rate 0.03, and gradient-norm clipping at 1. Adam uses betas 0.9/0.999 and epsilon `1e-8`. New runs start with unit gains; adaptation runs instead inherit source-trained gains. Optimization seeds change sample/minibatch order and any declared exploration, not initial anatomical weights or the fixed interfaces.

Budgets of 40, 120, and 200 epochs run in full. The 600-epoch maximum permits stopping after at least 200 epochs with validation patience 60. Validation selects balanced accuracy for classification and RMSE for regression; an improvement must exceed `1e-8`, so ties retain the earlier checkpoint. Epoch zero is eligible. Reported performance and saved inference weights belong to the selected checkpoint, not necessarily the last epoch.

## Global permutation inference and diagnostic controls

Five primary tests evaluate the anatomical rate model under the 40-epoch procedure, at partition/optimization seeds 1729/0. Each task has one observed procedure and 199 null procedures with permutation seeds 10000 through 10198. Complete target vectors are exchanged between equal-sized groups, preserving target multiplicities and the within-group target pattern. One permuted dataset is retained across all five folds. Label-dependent stratification and the entire fitting and selection procedure are repeated. Observed runs intentionally duplicate the same anatomical 40-epoch procedure used elsewhere.

The primary statistic is pooled OOF balanced accuracy for classification and negative RMSE for regression. R² remains descriptive. The minimum attainable unadjusted p-value is 0.005; Holm correction covers the five tasks. The [permutation protocol](permutation_protocol.md) defines the probability calculation and exchangeability assumptions. Its inference does not extend to other budgets, graphs or Ki-67.

Separate fixed training-target shuffles use seed 9701 and retain true validation and test targets. Their validation-selected results are diagnostic, not draws from the global null. Frozen controls keep initial parameters and record metrics at nominal epochs with no updates.

## Spiking dynamics and local learning

Eight LIF conditions cross thresholds 0.1/0.5, input amplitudes 2/4, and 24/48 steps, with fixed tonic current 0.5 and a temporal-mean readout. Two further terminal-readout conditions use visual or direct-KC inputs, threshold 0.1, input amplitude 2, and 40 steps, without tonic current. No threshold, current, or interface is learned. Several dynamic assumptions differ from the reference rate model, so this comparison does not isolate the effect of spikes alone.

Local plasticity selects only 837 existing KC-to-MBON edges. Its four regimes are visual baseline, direct-KC baseline, visual with stronger DAN pulses (amplitude 2, duration 6), and visual with exploration standard deviation 0.3. Other regimes use pulse amplitude 1, duration 3, exploration 0.15, and eligibility decay 0.8. The local learning rate is 0.05 except in zero-rate controls.

Each regime includes learning, zero learning rate, no feedback, and shuffled feedback. The shuffled condition reuses the trained partner's pulse values and count per epoch, changing their association with observations while retaining the corresponding task, partition, fold, order, and budget. Stored histories include activity, internal changes, DAN responses, and local diagnostics. Zero-feedback branches can skip eligibility evaluation; their recorded zeros do not demonstrate presynaptic silence. The [local rule](plasticity.md) defines the computation and its limits.

## Transfer within Iris

Separate lower- and upper-tail domains are defined by the median of sepal length, except when sepal length is the target, in which case sepal width is used. The threshold depends only on a retained input. Both tails include ties and can overlap at the median. Each domain uses partition seeds 1729/2718 and a common held-out target test.

| Displayed condition | Training data and initialization | Saved phase |
| --- | --- | --- |
| Source only | Source-domain data, initialized from anatomical gains | `source` |
| Source + target | Target-domain training data, initialized from selected source gains | `adapt` |
| Target only | Same target-domain training data, initialized from anatomical gains | `reset` |

Source training uses observations outside the selected target tail. All three conditions share the held-out target test and source-training scalers. Adaptation and target-only training use identical target training/validation arrays.

Each phase has a 120-epoch budget and its own validation-based selection. Adaptation receives 120 additional epochs after source experience; it does not have the same total training exposure as target-only training. Adaptation inherits the selected source gains exactly, including its initial predictions. The 20 triplets use small target holdouts of 15 to 19 observations. They are neither complete OOF evaluations nor an independent external dataset. The saved `reset` key describes target-only training and is unrelated to the transient activity-reset control.

## Saved results

Each run retains its configuration, partition/scaling data, selected checkpoint, initial/selected test predictions, controls and epoch history. Training and validation metrics cover epoch zero through the last executed epoch; Iris does not save a per-epoch test trajectory. The [evidence index](manuscript_evidence.md) locates campaign aggregates, permutation results and transfer triplets. The [computational-cost guide](full_graph_compute.md) defines the timing records.
