# Paper-anchored surrogate contract

The surrogate was frozen before controller experiments. At distance 15 it has 1,289 gates x 30 controls = 38,670 parameters.

It uses sparse local detector factors, frozen sensitivities/floors/curvatures, quadratic mismatch, sinusoidal and step drift, and distinct surrogate/controller/certification splits.

## Non-equivalences

- No Willow device, pulse stack, proprietary calibration state, detector graph, or control sensitivities.
- Frozen random local factors replace proprietary control-to-detector coefficients.
- Independent conditional Bernoulli detector counts omit unknown hardware correlations and leakage dynamics.
- The logical metric is a monotone declared risk proxy, not decoder-estimated hardware LER.
- Stim validates public surface-code topology where installed; it does not make the quadratic plant a circuit/pulse simulator.
- Wall-clock protocol projections are accounting fields, not measured hardware latency.
