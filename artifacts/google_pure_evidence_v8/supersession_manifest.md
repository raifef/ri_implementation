# Superseded v8 Evidence Artifacts

- **schema_version**: `"google-pure-evidence-v8-supersession.v1"`
- **legacy_artifacts_preserved**: `true`
- **legacy_artifacts_valid_for_new_claims**: `false`
- **prompt1_hash**: `"68c3d8f21d675a98ac09407432166ad09fc403069e9574db29a0f95b5a5ac483"`
- **artifact_hash**: `"7ae8af750b044a502dbb81a89bb38799eedeb81f6a9ce3af0f814ba847893981"`

## Records

| legacy_artifact | superseded_by | reason |
| --- | --- | --- |
| claim_registry.json | paper_claim_registry.json | old registry lacked required claim statuses and enforced anti-conflation |
| step_response/summary.json | step_response/results.json | old response normalized crossings by achieved motion |
| recovery/summary.json | recovery/results.json | old run did not explicitly spoil a policy |
