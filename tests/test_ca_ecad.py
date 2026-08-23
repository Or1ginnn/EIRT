import json
import os
import subprocess
import sys
import tempfile
import unittest


from search_r1.llm_agent.ca_ecad import (
    ACTION_ANSWER,
    ACTION_INVALID,
    ACTION_SEARCH,
    INVALID_SEGMENT_ID,
    build_policy_segment_ids,
    canonical_document_id,
    credit_values_by_segment,
    compute_ca_ecad_credits,
    compute_peer_diagnostics,
    contiguous_policy_spans,
    ordered_document_ids,
    summarize_absolute_credit,
    top1_peer_key,
    validate_segment_partition,
    write_json,
    write_jsonl,
)


class DocumentIdentityTest(unittest.TestCase):

    def test_ordered_ids_preserve_retrieval_rank(self):
        result = [
            {'document': {'id': 9, 'title': 'third'}},
            {'document': {'id': '2', 'title': 'first'}},
            {'document': {'id': None, 'title': 'missing'}},
        ]
        self.assertEqual(ordered_document_ids(result), ('9', '2', None))

    def test_document_ids_do_not_fall_back_to_title(self):
        self.assertEqual(ordered_document_ids([{'document': {'title': 'Hamlet'}}]), (None,))
        self.assertIsNone(canonical_document_id(True))
        self.assertIsNone(top1_peer_key('q', 1, [None, '2']))
        self.assertEqual(top1_peer_key('q', 2, ['9', '2']), ('q', 2, '9'))


class TokenOwnershipTest(unittest.TestCase):

    def test_invalid_turn_folds_into_next_search(self):
        generation_turn_ids = [1, 1, 0, 2, 2, 0, 3, 3, -1]
        policy_mask = [1, 1, 0, 1, 1, 0, 1, 1, 0]
        actions = [ACTION_INVALID, ACTION_SEARCH, ACTION_ANSWER, -1]
        segments = build_policy_segment_ids(generation_turn_ids, actions, policy_mask)
        self.assertEqual(segments, [1, 1, -1, 1, 1, -1, 2, 2, -1])
        validate_segment_partition(segments, policy_mask, search_count=1)
        self.assertEqual(
            contiguous_policy_spans(segments, policy_mask),
            {'1': [[0, 2], [3, 5]], '2': [[6, 8]]},
        )

    def test_no_search_assigns_all_model_tokens_to_utilization(self):
        generation_turn_ids = [1, 1, 0, 2, 2]
        policy_mask = [1, 1, 0, 1, 1]
        actions = [ACTION_INVALID, ACTION_ANSWER]
        segments = build_policy_segment_ids(generation_turn_ids, actions, policy_mask)
        self.assertEqual(segments, [1, 1, INVALID_SEGMENT_ID, 1, 1])
        validate_segment_partition(segments, policy_mask, search_count=0)

    def test_environment_token_cannot_receive_credit(self):
        with self.assertRaises(ValueError):
            validate_segment_partition([1, 1], [0, 1], search_count=0)


class CreditConstructionTest(unittest.TestCase):

    def test_peer_conditioning_and_telescoping_conservation(self):
        result = compute_ca_ecad_credits(
            group_uids=['q', 'q', 'q'],
            rewards=[0.0, 1.0, 0.0],
            search_histories=[[['A', 'x']], [['A', 'y']], [['B', 'z']]],
            success_prior=0.5,
        )
        failed_with_successful_peer = result['records'][0]
        self.assertAlmostEqual(failed_with_successful_peer['baseline'], 0.5)
        self.assertEqual(failed_with_successful_peer['turns'][0]['peer_support'], 1)
        self.assertEqual(failed_with_successful_peer['turns'][0]['peer_successes'], 1.0)
        self.assertGreater(failed_with_successful_peer['turns'][0]['acquisition_advantage'], 0.0)
        self.assertLess(failed_with_successful_peer['utilization_advantage'], 0.0)
        self.assertLess(result['metrics']['ca_ecad/phase2/max_conservation_error'], 1e-12)

    def test_peer_free_later_turn_has_zero_increment(self):
        result = compute_ca_ecad_credits(
            group_uids=['q'],
            rewards=[1.0],
            search_histories=[[['A'], ['B']]],
            success_prior=0.4,
        )
        turns = result['records'][0]['turns']
        self.assertGreater(turns[0]['acquisition_advantage'], 0.0)
        self.assertAlmostEqual(turns[1]['acquisition_advantage'], 0.0)

    def test_no_search_uses_outcome_minus_loo_baseline(self):
        result = compute_ca_ecad_credits(
            group_uids=['q', 'q'],
            rewards=[1.0, 0.0],
            search_histories=[[], []],
            success_prior=0.25,
        )
        first = result['records'][0]
        self.assertEqual(first['turns'], [])
        self.assertAlmostEqual(first['no_search_advantage'], first['reward'] - first['baseline'])

    def test_non_binary_reward_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_ca_ecad_credits(['q'], [0.1], [[]], success_prior=0.5)

    def test_credit_values_follow_token_segment_ownership(self):
        result = compute_ca_ecad_credits(
            group_uids=['q', 'q'],
            rewards=[1.0, 0.0],
            search_histories=[[['A'], ['B']], [['A'], ['C']]],
            success_prior=0.4,
        )
        record = result['records'][0]
        values = credit_values_by_segment(record)
        self.assertEqual(set(values), {1, 2, 3})
        self.assertAlmostEqual(
            sum(values.values()),
            record['reward'] - record['baseline'],
        )

    def test_no_search_credit_has_only_utilization_segment(self):
        result = compute_ca_ecad_credits(
            group_uids=['q', 'q'],
            rewards=[0.0, 1.0],
            search_histories=[[], []],
            success_prior=0.5,
        )
        values = credit_values_by_segment(result['records'][0])
        self.assertEqual(set(values), {1})
        self.assertAlmostEqual(values[1], result['records'][0]['no_search_advantage'])


class PeerDiagnosticsTest(unittest.TestCase):

    def test_peer_stats_use_uid_turn_and_top1(self):
        metrics = compute_peer_diagnostics(
            group_uids=['q', 'q', 'q', 'other'],
            rewards=[0.0, 1.0, 0.0, 1.0],
            search_histories=[
                [['A', 'B', 'C']],
                [['A', 'B', 'C']],
                [['D', 'B', 'C']],
                [['A', 'B', 'C']],
            ],
        )
        self.assertEqual(metrics['ca_ecad/phase2/top1_peer_supported_turn_count'], 2.0)
        self.assertEqual(metrics['ca_ecad/phase2/mixed_outcome_peer_turn_count'], 2.0)
        self.assertAlmostEqual(metrics['ca_ecad/phase2/ordered_top3_match_given_top1_match'], 1.0)
        self.assertEqual(metrics['ca_ecad/phase2/unique_top1_document_count'], 2.0)

    def test_all_zero_repeated_mode_is_visible(self):
        metrics = compute_peer_diagnostics(
            group_uids=['q', 'q', 'q'],
            rewards=[0.0, 0.0, 0.0],
            search_histories=[[['A']], [['A']], [['B']]],
        )
        self.assertEqual(metrics['ca_ecad/phase2/all_zero_group_count'], 1.0)
        self.assertEqual(metrics['ca_ecad/phase2/all_zero_repeated_mode_group_count'], 1.0)
        self.assertEqual(metrics['ca_ecad/phase2/all_zero_repeated_mode_rate'], 1.0)

    def test_mode_diversity_is_distinct_from_peer_support(self):
        metrics = compute_peer_diagnostics(
            group_uids=['q', 'q', 'q', 'q', 'q'],
            rewards=[0.0, 0.0, 1.0, 0.0, 1.0],
            search_histories=[[['A']], [['A']], [['A']], [['A']], [['B']]],
        )
        self.assertAlmostEqual(metrics['ca_ecad/phase2/distinct_top1_mode_mean/k_1'], 2.0)
        self.assertAlmostEqual(metrics['ca_ecad/phase2/all_same_top1_mode_rate/k_1'], 0.0)
        self.assertAlmostEqual(metrics['ca_ecad/phase2/multiple_top1_mode_rate/k_1'], 1.0)

    def test_missing_ids_are_reported_not_grouped(self):
        metrics = compute_peer_diagnostics(
            group_uids=['q', 'q'],
            rewards=[0.0, 1.0],
            search_histories=[[[None, 'B']], [[None, 'B']]],
        )
        self.assertEqual(metrics['ca_ecad/phase2/top1_document_id_presence_rate'], 0.0)
        self.assertEqual(metrics['ca_ecad/phase2/top1_peer_supported_turn_count'], 0.0)
        self.assertEqual(metrics['ca_ecad/phase2/all_document_id_presence_rate'], 0.5)


class CreditMagnitudeTest(unittest.TestCase):

    def test_absolute_credit_summary_does_not_cancel_signs(self):
        summary = summarize_absolute_credit([-0.02, 0.03, -0.01, 0.00])
        self.assertAlmostEqual(summary['mean_abs'], 0.015)
        self.assertAlmostEqual(summary['rate_abs_gt_0_01'], 0.5)
        self.assertAlmostEqual(summary['rate_abs_gt_0_05'], 0.0)


class ArtifactWriterTest(unittest.TestCase):

    def test_atomic_json_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = os.path.join(directory, 'summary.json')
            jsonl_path = os.path.join(directory, 'records.jsonl')
            write_json(json_path, {'ok': True})
            write_jsonl(jsonl_path, [{'row': 1}, {'row': 2}])
            with open(json_path, 'r', encoding='utf-8') as handle:
                self.assertEqual(json.load(handle), {'ok': True})
            with open(jsonl_path, 'r', encoding='utf-8') as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(rows, [{'row': 1}, {'row': 2}])

    def test_phase2_analyzer_builds_credits_only_after_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout_path = os.path.join(directory, 'phase2_step_000001_rollouts.jsonl')
            write_jsonl(rollout_path, [
                {'group_uid': '1:p1', 'reward': 0.0, 'ordered_search_document_ids': [['A', 'x']]},
                {'group_uid': '1:p1', 'reward': 1.0, 'ordered_search_document_ids': [['A', 'y']]},
                {'group_uid': '1:p2', 'reward': 0.0, 'ordered_search_document_ids': [['B', 'x']]},
                {'group_uid': '1:p2', 'reward': 0.0, 'ordered_search_document_ids': [['B', 'y']]},
            ])
            script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'scripts',
                'diagnostics',
                'analyze_ca_ecad_phase2.py',
            )
            subprocess.run(
                [
                    sys.executable,
                    script,
                    '--input-dir', directory,
                    '--min-calibration-prompts', '2',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with open(os.path.join(directory, 'phase2_analysis.json'), 'r', encoding='utf-8') as handle:
                analysis = json.load(handle)
            self.assertTrue(analysis['ready_for_phase3_implementation'])
            self.assertEqual(analysis['rollout_count'], 4)
            with open(os.path.join(directory, 'phase2_credit_records.jsonl'), 'r', encoding='utf-8') as handle:
                self.assertEqual(len(handle.readlines()), 4)

    def test_signal_analyzer_reports_credit_magnitude_and_mode_diversity(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout_path = os.path.join(directory, 'phase2_step_000001_rollouts.jsonl')
            state_path = os.path.join(directory, 'global_step_100.json')
            write_jsonl(rollout_path, [
                {
                    'group_uid': '1:p1', 'reward': 0.0,
                    'ordered_search_document_ids': [['A', 'x']],
                    'policy_segment_spans': {'1': [[0, 2]], '2': [[2, 5]]},
                },
                {
                    'group_uid': '1:p1', 'reward': 1.0,
                    'ordered_search_document_ids': [['B', 'y']],
                    'policy_segment_spans': {'1': [[0, 4]], '2': [[4, 5]]},
                },
            ])
            write_json(state_path, {
                'version': 1,
                'success_prior': 0.5,
                'hyperparameters': {
                    'alpha': 2.0, 'eta': 0.25, 'kappa': 1.0,
                    'success_prior_rho': 0.01, 'mode_balance_gamma': 0.0,
                },
            })
            script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'scripts', 'diagnostics', 'analyze_ca_ecad_signal.py',
            )
            subprocess.run(
                [
                    sys.executable, script, '--input-dir', directory,
                    '--prior-state-path', state_path,
                ],
                check=True, capture_output=True, text=True,
            )
            with open(os.path.join(directory, 'ca_ecad_signal_audit.json'), 'r', encoding='utf-8') as handle:
                audit = json.load(handle)
            self.assertEqual(audit['rollout_count'], 2)
            self.assertGreater(audit['acquisition_by_search_round']['k_1']['mean_abs'], 0.0)
            self.assertEqual(
                audit['environment_mode_diversity_by_search_round']['k_1']['distinct_top1_mode_mean'], 2.0,
            )
            self.assertAlmostEqual(
                audit['absolute_policy_signal_mass']['acquisition_share'] +
                audit['absolute_policy_signal_mass']['utilization_share'],
                1.0,
            )


if __name__ == '__main__':
    unittest.main()
