"""CPU-only runner contract tests. No torch, simulator, or model import."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('worker', Path(__file__).with_name('vtcg_factorial_episode.py'))
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class RunnerTests(unittest.TestCase):
    def test_import_is_lightweight(self):
        self.assertNotIn('torch', sys.modules)

    def test_ragged_optional_arrays(self):
        records = []
        for index, count in enumerate((16, 48, 64)):
            row = {key: np.ones((7, 256), np.float32) for key in ('positive', 'negative', 'guided')}
            row.update(selected_token_ids=np.arange(count, dtype=np.int16),
                       selected_mask=np.ones(256, np.uint8),
                       step_index=np.asarray(index, np.int32))
            if index < 2:
                row['selected_source_features'] = np.ones((count, 8), np.float32)
            records.append(row)
        arrays = worker.serialize_records(records)
        self.assertEqual(arrays['positive_logits'].shape, (3, 7, 256))
        self.assertEqual(arrays['selected_mask'].dtype, np.uint8)
        self.assertEqual(arrays['step_0001__selected_token_ids'].shape, (48,))
        self.assertNotIn('step_0002__selected_source_features', arrays)
        self.assertTrue(all(not v.dtype.hasobject for v in arrays.values()))
        records[0]['positive'] = records[0]['positive'].astype(np.float16)
        with self.assertRaises(RuntimeError):
            worker.serialize_records(records)

    def test_atomic_no_clobber_and_claim(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'meta.json'
            worker.atomic_json(path, {'first': True}, exclusive=True)
            with self.assertRaises(FileExistsError):
                worker.atomic_json(path, {'first': False}, exclusive=True)
            self.assertEqual(json.loads(path.read_text()), {'first': True})
            with worker.claim(Path(root) / 'claim.lock'):
                with self.assertRaises(RuntimeError):
                    with worker.claim(Path(root) / 'claim.lock'):
                        pass

    def test_frozen_phase(self):
        path = worker.ROOT / 'artifacts/vtcg_factorial_controls_v1'
        args = worker.parse_args(['--artifact', str(path), '--task', worker.TASKS[0],
            '--seed', '500', '--arm', 'attn_topp', '--gpu', '1', '--snapshot-mode', 'run',
            '--phase', 'formal', '--run-config-id', 'VTCG_FACTORIAL_CONTROLS_V1-formal'])
        lock, _ = worker.validate_lock(args)
        self.assertIn(500, lock['formal_seeds'])
        args.run_config_id = 'wrong'
        with self.assertRaises(ValueError):
            worker.validate_lock(args)

    def test_audit_rejects_wrong_constructor(self):
        trace = [{'arm': 'vanilla', 'feature_equal': True, 'non_target_bit_identical': True,
                  'counterfactual_constructor': 'none', 'guided_prefix': True,
                  'attention_layers': [11], 'lambda': .5, 'beta': 0,
                  'intervention_location': 'projector', 'positive_token_ids': list(range(7)),
                  'final_token_ids': list(range(7))}]
        self.assertTrue(worker.audit_trace(trace, 'vanilla')['technical_pass'])
        trace[0]['counterfactual_constructor'] = 'harmonic'
        with self.assertRaises(RuntimeError):
            worker.audit_trace(trace, 'vanilla')


if __name__ == '__main__':
    unittest.main()
