"""Consistency checks for the experiment folders. CPU only; no model is loaded."""
import hashlib
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EarlyRunFiles(unittest.TestCase):
    def test_grpo_trainer_copy_matches_derivation(self):
        source = (ROOT / 'tabcrew_rl' / 'train.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('if len(tokenizer.encode(prompt))>2048:'), 1)
        derived = source.replace('if len(tokenizer.encode(prompt))>2048:', 'if len(tokenizer.encode(prompt))>4096:')
        derived = derived.replace('from .', 'from tabcrew_rl.')
        self.assertEqual((EXPERIMENTS / 'early-705' / 'grpo_train.py').read_text(encoding='utf-8'), derived)

    def test_grpo_config_matches_derivation(self):
        config = json.loads((ROOT / 'configs' / 'grpo-after-sft-pilot.json').read_text())
        config.update(max_steps=16, max_wall_seconds=900, max_tabs=30, max_completion_length=512,
                      note='705-example run: after one complete SFT epoch; frozen train only; beta=0; '
                           'stop on three equal-reward groups')
        self.assertEqual(json.loads((EXPERIMENTS / 'early-705' / 'grpo-config.json').read_text()), config)


class ContractRepairPlan(unittest.TestCase):
    def test_pinned_code_hashes_match_files(self):
        plan = json.loads((EXPERIMENTS / 'contract-repair' / 'training-plan.json').read_text())
        for name, expected in plan['code_sha256'].items():
            self.assertEqual(sha(EXPERIMENTS / 'contract-repair' / name), expected, name)

    def test_batching_is_shared(self):
        self.assertEqual(sha(EXPERIMENTS / 'contract-repair' / 'batching.py'),
                         sha(EXPERIMENTS / 'mixed-sft' / 'batching.py'))


class CompletionBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batching = load_module(EXPERIMENTS / 'mixed-sft' / 'batching.py', 'mixed_sft_batching')

    def test_left_padding_and_completion_labels(self):
        batch = self.batching.collate_completion_batch(
            [('a', [5, 6, 7], [8, 2]), ('b', [5], [9, 9, 2])], pad_token_id=0)
        self.assertEqual(batch['episode_ids'], ['a', 'b'])
        self.assertEqual(batch['input_ids'].tolist(), [[5, 6, 7, 8, 2], [0, 5, 9, 9, 2]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        self.assertEqual(batch['position_ids'].tolist(), [[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]])
        self.assertEqual(batch['labels'].tolist(), [[-100, 8, 2], [9, 9, 2]])
        self.assertEqual(batch['target_lengths'].tolist(), [2, 3])
        self.assertEqual(batch['logits_to_keep'], 4)

    def test_rejects_invalid_batches(self):
        collate = self.batching.collate_completion_batch
        with self.assertRaises(ValueError):
            collate([], pad_token_id=0)
        with self.assertRaises(ValueError):
            collate([('a', [1], [2])] * 5, pad_token_id=0)
        with self.assertRaises(ValueError):
            collate([('a', [1], [])], pad_token_id=0)
        with self.assertRaises(ValueError):
            collate([('a', [1], [2])], pad_token_id=-1)


if __name__ == '__main__':
    unittest.main()
