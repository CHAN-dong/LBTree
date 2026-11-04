"""Protocol failures must be detected before model training or parameter selection."""
import json
from pathlib import Path
import tempfile
import unittest
from reproducibility import inspect, workload_stats, DEFAULT_CONFIG


class ProtocolChecks(unittest.TestCase):
    def test_disjoint_rejects_reordered_duplicate(self):
        from reproducibility import read_sets
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'q.csv'
            p.write_text('Title\nb a\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                workload_stats([('a','b')], list(read_sets(p)), 'disjoint')

    def test_shared_is_order_sensitive(self):
        with self.assertRaisesRegex(ValueError, 'identical ordered'):
            workload_stats([('a',),('b',)], [('b',),('a',)], 'shared')
        r = workload_stats([('a',)], [('a',)], 'shared')
        self.assertEqual(r['normalized_overlap'], 1)

    def test_validation_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'validation'):
            workload_stats([('a',)], [('b',)], 'disjoint', [('b',)])

    def test_defaults_do_not_invent_validation(self):
        r = workload_stats([('a',)], [('b',)], 'disjoint')
        self.assertIsNone(r['validation_queries'])
        self.assertEqual(r['parameter_selection'], 'fixed_defaults_no_selection')

    def fixture(self, td):
        paths = [Path(td)/f'{k}.csv' for k in ('objects','train','eval')]
        for p, text in zip(paths, ('a b\nb c\nc', 'a b', 'c')):
            p.write_text('Title\n'+text+'\n', encoding='utf-8')
        return paths

    def test_observed_counts_and_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.fixture(td)
            r = inspect(*p)
            self.assertEqual((r['objects'], r['active_keywords'], r['keyword_object_pairs']), (3,3,5))
            self.assertEqual(r['normalized_overlap'], 0)
            self.assertEqual(len(r['inputs']['objects']['sha256']), 64)

    def test_count_mismatch_not_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.fixture(td)
            r = inspect(*p, dataset='DBLP')
            self.assertFalse(r['paper_counts_matched'])
            with self.assertRaisesRegex(ValueError, 'differs'):
                inspect(*p, dataset='DBLP', enforce_counts=True)

    def test_unknown_training_term_rejected_but_eval_retained(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.fixture(td)
            p[2].write_text('Title\nunknown\n', encoding='utf-8')
            self.assertEqual(inspect(*p)['evaluation_oov_query_count'], 1)
            p[1].write_text('Title\nunknown\n', encoding='utf-8')
            p[2].write_text('Title\nc\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'construction terms'):
                inspect(*p)

    def test_six_baselines_and_exact_grid(self):
        cfg = json.loads(DEFAULT_CONFIG.read_text())
        b = {x['name']:x for x in cfg['baselines']}
        self.assertEqual(set(b), {'Bitmap','Roaring','TEB','BigSet','MILC','Bitlist'})
        self.assertEqual(b['TEB']['grid']['rank_block_bits'], [128,256,512,1024])
        self.assertEqual(len(b['BigSet']['grid']['k'])*len(b['BigSet']['grid']['c']), 9)
        self.assertEqual(b['MILC']['fidelity'], 'core_only')

    def test_distinct_profiles_and_synthetic_rules(self):
        cfg = json.loads(DEFAULT_CONFIG.read_text())
        s = cfg['datasets']['SYN']
        self.assertEqual(s['paper_claim']['objects'], s['historical_shared']['objects'])
        self.assertNotEqual(s['paper_claim']['generator'], s['historical_shared']['generator'])
        self.assertEqual(cfg['profiles']['historical_shared']['workload_policy'], 'shared')


if __name__ == '__main__':
    unittest.main()
