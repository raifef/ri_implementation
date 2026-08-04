# Retained compact inputs

These JSON files are small provenance inputs needed to verify the historical v5/v6 headline snapshots, the direct public-data replay, and the pre-repair Figure-5 protocol identity. They were copied from earlier Google-only executions and are explicitly marked as retained historical inputs, not new executions or final claim evidence.

`google-rl-bootstrap` copies them into `artifacts/` only when the corresponding target is absent, then regenerates the controller resolution, compact v8 diagnostics, and smoke evidence with the standalone source hashes.
