# Figure 5 Source Contract

> Paper-anchored synthetic reproduction; not Google's proprietary simulator.

- **5a.question** — `EXPLICITLY_SPECIFIED` — main Fig. 5a; SI VI.A
- **5a.axes** — `EXPLICITLY_SPECIFIED` — main Fig. 5a; SI Fig. S8c
- **5a.normalization** — `EXPLICITLY_SPECIFIED` — SI VI.A
- **5a.budget** — `EXPLICITLY_SPECIFIED` — SI VI.A
- **5a.frequency_anchor** — `EXPLICITLY_SPECIFIED` — main text and SI VI.A
- **5a.exact_frequency_grid** — `SYNTHETIC_REPRODUCTION_CHOICE` — public source does not enumerate the grid
- **5a.exact_entropy_grid** — `SYNTHETIC_REPRODUCTION_CHOICE` — the public source illustrates values but does not publish the Fig. 5a grid
- **5b.question** — `EXPLICITLY_SPECIFIED` — main Fig. 5b-c; SI VI.B
- **5b.distances** — `EXPLICITLY_SPECIFIED` — SI VI.B
- **5b.memory_cycles** — `EXPLICITLY_SPECIFIED` — SI VI.B
- **5b.control_count** — `EXPLICITLY_SPECIFIED` — SI Eq. 7
- **5b.d15_p30_controls** — `DERIVED_FROM_EXPLICIT_SOURCE` — SI Eq. 7 and Table I
- **5b.irreducible_floor** — `EXPLICITLY_SPECIFIED` — main Fig. 5b and SI VI.B
- **5b.exact_candidate_budget** — `NOT_PUBLICLY_SPECIFIED` — SI VI.B does not enumerate candidates/cycles/epochs for the scaling run
- **5b.proprietary_simulator** — `NOT_PUBLICLY_SPECIFIED` — main Code availability
- **5c.axes** — `EXPLICITLY_SPECIFIED` — main Fig. 5c caption; SI Eq. 8
- **5c.parameters_per_gate** — `EXPLICITLY_SPECIFIED` — main Fig. 5c
- **5c.fit** — `DERIVED_FROM_EXPLICIT_SOURCE` — SI Eq. 8
- **5c.derivative_estimator_and_fit_window** — `SYNTHETIC_REPRODUCTION_CHOICE` — public source states point estimates and linear fits but not numerical estimator details
- **shared.seed_cohorts** — `SYNTHETIC_REPRODUCTION_CHOICE` — public simulation seeds are not specified
- **shared.synthetic_plant** — `SYNTHETIC_REPRODUCTION_CHOICE` — Google simulator/code and full hyperparameters are proprietary
