import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from hdfa_rl_suite.evaluation.scalability import (
    PaperProtocol,
    PipelineCheckpointError,
    ScalabilityConfig,
    ScalabilityRunner,
    normalized_improvement,
    paper_control_parameters,
    physical_qubits,
    time_reduced_detector_factors,
)
from hdfa_rl_suite.evaluation.scalability_artifacts import write_scalability_artifacts
from hdfa_rl_suite.stage6 import ExplorationBudget, GaussianResidualPolicy, ResidualRLController


class PaperMatchedScalabilityTests(unittest.TestCase):
    def test_published_structural_anchors(self):
        self.assertEqual(paper_control_parameters(15, 30), 38_670)
        self.assertEqual(physical_qubits(15), 449)
        self.assertEqual(time_reduced_detector_factors(5), PaperProtocol().distance_5_reward_components)

    def test_normalized_improvement_has_paper_endpoints(self):
        self.assertEqual(normalized_improvement(10., 10., 2.), 0.)
        self.assertEqual(normalized_improvement(2., 10., 2.), 1.)
        self.assertLess(normalized_improvement(12., 10., 2.), 0.)

    def test_smoke_report_is_reproducible_and_fitted(self):
        config = ScalabilityConfig.for_profile("smoke")
        first, second = ScalabilityRunner(config).run(), ScalabilityRunner(config).run()
        self.assertEqual(first.report_hash, second.report_hash)
        self.assertTrue(first.scaling)
        self.assertTrue(first.steerability)
        self.assertTrue(all(fit.gamma > 0 for fit in first.fits))
        self.assertFalse(any(gate.status == "fail" for gate in first.gates))

    def test_artifacts_are_parseable_and_checksummed(self):
        report = ScalabilityRunner(ScalabilityConfig(
            distances=(3,), parameters_per_gate=(1,), epochs=4, seeds=(7,),
            steering_frequencies=(1e-3, 1e-2), entropy_regularizations=(1e-3, 1e-2),
            steering_epochs=20,
        )).run()
        with tempfile.TemporaryDirectory() as directory:
            paths = write_scalability_artifacts(report, Path(directory))
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["report_hash"], report.report_hash)
            for name in ("fig5a", "fig5b", "fig5c", "resource_plot"):
                ET.parse(paths[name])
                self.assertIn(name, manifest["artifacts"])

    def test_stage6_identity_policy_storage_and_sampling_are_linear(self):
        controls = tuple(f"u{index}" for index in range(2_000))
        policy = GaussianResidualPolicy.full_control_baseline(controls, .01)
        self.assertEqual(sum(len(row) for row in policy.covariance.values()), len(controls))
        graph = {f"d{index}": controls[index:index+2] for index in range(0, len(controls), 2)}
        controller = ResidualRLController(policy, graph, ExplorationBudget(100., 1_000.))
        sample = controller._sample(controls)
        self.assertEqual(len(sample), len(controls))

    def test_full_profile_uses_bounded_independent_workers(self):
        self.assertEqual(ScalabilityConfig.for_profile("full").pipeline_workers, 8)
        with self.assertRaises(ValueError):
            ScalabilityConfig(pipeline_workers=0)

    def test_spawned_pipeline_probe_has_matched_stationary_onsets(self):
        config = ScalabilityConfig(
            distances=(3,), parameters_per_gate=(1,), epochs=2, seeds=(7, 19),
            steering_frequencies=(1e-3, 1e-2),
            entropy_regularizations=(1e-3, 1e-2), steering_epochs=2,
            run_pipeline_probe=True, pipeline_distances=(3,), pipeline_epochs=1,
            pipeline_cycles_per_interval=4, pipeline_candidate_cycles=2,
            pipeline_candidates=4, pipeline_workers=2,
            pipeline_bootstrap_characterization_shots=64,
            pipeline_bootstrap_validation_cycles=64,
            pipeline_bootstrap_target_stddev=.07,
            pipeline_bootstrap_qec_rate_limit=.20,
            pipeline_baseline_cycles=32,
        )
        points, failures = ScalabilityRunner(config)._pipeline_probe()
        self.assertFalse(failures)
        self.assertEqual(len(points), 4)
        for seed in config.seeds:
            rows = [row for row in points if row.seed == seed]
            self.assertEqual(len({row.pre_disturbance_observation_hash for row in rows}), 1)
            self.assertEqual(len({row.disturbance_realization_id for row in rows}), 1)
            self.assertEqual(len({row.disturbance_epoch_s for row in rows}), 1)
            self.assertEqual(len({row.bootstrap_execution_id for row in rows}), 1)
            self.assertAlmostEqual(sum(row.bootstrap_execution_share for row in rows), 1.0)
            self.assertTrue(all(row.peak_process_memory_bytes > 0 for row in rows))
            self.assertTrue(all(row.peak_incremental_process_memory_bytes >= 0
                                for row in rows))
            self.assertTrue(all(row.memory_measurement == "sampled_process_resident_set"
                                for row in rows))
            self.assertTrue(all(row.condition_process_isolation
                                == "fresh_process_per_condition" for row in rows))

    def test_condition_checkpoints_resume_with_resized_worker_pool(self):
        config = ScalabilityConfig(
            distances=(3,), parameters_per_gate=(1,), epochs=2, seeds=(7, 19),
            steering_frequencies=(1e-3, 1e-2),
            entropy_regularizations=(1e-3, 1e-2), steering_epochs=2,
            run_pipeline_probe=True, pipeline_distances=(3,), pipeline_epochs=1,
            pipeline_cycles_per_interval=4, pipeline_candidate_cycles=2,
            pipeline_candidates=4, pipeline_workers=1,
            pipeline_bootstrap_characterization_shots=64,
            pipeline_bootstrap_validation_cycles=64,
            pipeline_bootstrap_target_stddev=.07,
            pipeline_bootstrap_qec_rate_limit=.20,
            pipeline_baseline_cycles=32,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory)
            first = ScalabilityRunner(
                config, checkpoint_directory=checkpoints).run()
            self.assertEqual(len(tuple(checkpoints.glob("*.json"))), 2)
            resized = ScalabilityRunner(
                replace(config, pipeline_workers=4),
                checkpoint_directory=checkpoints, resume=True).run()
            first_scientific = [
                (row.method, row.seed, row.detector_event_rate,
                 row.pre_disturbance_observation_hash, row.disturbance_realization_id)
                for row in first.pipeline_probe
            ]
            resized_scientific = [
                (row.method, row.seed, row.detector_event_rate,
                 row.pre_disturbance_observation_hash, row.disturbance_realization_id)
                for row in resized.pipeline_probe
            ]
            self.assertEqual(first_scientific, resized_scientific)
            self.assertEqual({row.worker_concurrency for row in resized.pipeline_probe}, {1})

    def test_corrupt_checkpoint_is_never_silently_reused(self):
        config = ScalabilityConfig(
            distances=(3,), parameters_per_gate=(1,), epochs=2, seeds=(7,),
            steering_frequencies=(1e-3, 1e-2),
            entropy_regularizations=(1e-3, 1e-2), steering_epochs=2,
            run_pipeline_probe=True, pipeline_distances=(3,), pipeline_epochs=1,
            pipeline_workers=1,
            pipeline_bootstrap_characterization_shots=64,
            pipeline_bootstrap_validation_cycles=64,
            pipeline_bootstrap_target_stddev=.07,
            pipeline_bootstrap_qec_rate_limit=.20,
            pipeline_baseline_cycles=32,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory)
            ScalabilityRunner(config, checkpoint_directory=checkpoints).run()
            path = next(checkpoints.glob("*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["result_hash"] = "corrupt"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PipelineCheckpointError):
                ScalabilityRunner(
                    config, checkpoint_directory=checkpoints, resume=True).run()


if __name__ == "__main__":
    unittest.main()
