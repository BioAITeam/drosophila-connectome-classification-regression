# Local plasticity with perturbation and feedback

[`LocalRewardLearner`](<../3 Connectome/flybrain/plasticity.py>) implements a computational three-factor learning hypothesis. It is a complementary Iris experiment, distinct from the supervised backpropagation used by the reference model and BRAP-Net. PAM and PPL1 feedback valences are modeling assumptions, not a measured plasticity rule.

## One learning trial

1. Present the four input intensities through the same fixed interface used by the supervised condition. Add Gaussian exploration current to the postsynaptic cells of existing plastic edges. This perturbation is constant during the presentation and does not receive target labels.
2. Simulate activity for the declared duration. Classification selects the largest mean among three fixed output populations; regression uses the fixed difference between the means of populations G0 and G1.
3. During training, compare the response with the target. Classification returns +1 for a correct class and -1 otherwise. Regression compares the squared errors of perturbed and unperturbed responses of the same current model before updating; the reduction is clipped to `[-1, 1]`. This reference is not a learned critic.
4. Positive feedback applies current to the annotated reward DAN population, and negative feedback to the punishment population. Two simulations start from exactly the same post-stimulus state: one receives the pulse and the other does not. Their positive activity difference isolates the pulse-evoked DAN response in this contrast.
5. Project that difference only through existing DAN-to-postsynaptic contacts. Modulatory weights are original contact counts divided by the total incoming DAN contacts of that cell. Positive reward and negative punishment valences are fixed hypotheses. Initialization requires both DAN populations and a valid direct route for each valence.
6. Calculate a bounded local efficacy update. The next trial clears transient states but retains the internal gains.

The DAN pulse enters the simulator's neuronal equations. The modulatory projection, normalization, and sign are additional assumptions applied to measured edges; anatomical counts are not measurements of dopamine potency. DAN fast-transmission signs remain zero. Their simulated activity and contacts can still participate in the explicitly modeled modulatory rule.

## Implemented update

For an existing selected edge `e = pre -> post`, eligibility is a weighted temporal mean involving previous presynaptic activity and the exploration applied to its postsynaptic cell:

```text
q_t is proportional to decay**(T - 1 - t), normalized over T steps
eligibility_e = sum(t=1..T-1, q_t * activity_pre(t-1) * noise_post / sigma**2)
delta_e = learning_rate * base_weight_e * mean_batch(eligibility_e * modulation_post)
delta_e = clip(delta_e, -0.05, 0.05)
```

`delta_e` changes the effective gain `g_e`, not the anatomical contact count. The base-weight factor includes normalization and presynaptic sign, preserving the direction of a gain change on an inhibitory connection. Because the perturbation enters before `tanh` and persists across recurrent steps, the estimate is a local perturbation heuristic rather than a claimed unbiased loss gradient. It uses no backpropagation, learned critic, or trainable readout. The current rule supports rate dynamics and explicitly rejects LIF dynamics.

Only edges whose postsynaptic cells receive modeled modulation can change. Edges with zero effective base weight also remain unchanged. Positions with exactly zero efficacy updates preserve the stored `gain` parameter exactly; a sigmoid-to-logit round trip must not create numerical changes mistaken for learning. Zero feedback or zero pulse amplitude changes no gains.

## Training API and controls

With a rate model, scaled data and a random generator already defined:

```python
from flybrain.plasticity import LocalRewardLearner

learner = LocalRewardLearner(model, learning_rate=0.05)
statistics = learner.step(x_train, y_train, generator=rng)
test_predictions = learner.predict(x_test)
test_values = learner.predict(x_test, task="regression")
```

The [local regimes](round4_protocol.md#spiking-dynamics-and-local-learning) use this learning rate except in zero-rate controls.

`step(..., update=False)` uses deterministic inference without exploration, pulses, or updates. Labels only affect the feedback statistic returned by that call, not its prediction. A frozen-training control that retains paired exploration and pulses instead uses `learning_rate=0`, `update=True`, and the same seeds. `reward_override` supplies declared zero or shuffled feedback.

In the paper, local learning selects 837 existing KC-to-MBON edges, whereas supervised subgraph learning selects all 40,755 retained edges. Learning signals and the number and location of adaptive edges therefore change together; their comparison does not isolate the learning rule alone.

Saved diagnostics include changed-edge counts, update norms, evoked responses in both DAN populations, and the fraction of candidate postsynaptic cells receiving reward, punishment, or both contact types. Coverage alone does not establish that every class can be learned. If the implementation skips eligibility calculations for a zero-feedback branch, a saved zero means that this diagnostic was not evaluated; it is not evidence of silent presynaptic activity.

The [evaluation guide](evaluation.md#controls-and-their-interpretation) defines the controls used to assess whether gain changes improve held-out prediction.
