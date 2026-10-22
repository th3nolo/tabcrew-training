import copy,json,pathlib,tempfile,unittest
from tabcrew_rl.compare import compare

class ComparisonTests(unittest.TestCase):
    def create(self,root):
        summary={'prompt_version':'a','reward_version':'b','decoding':{'cap':256},'dataset_manifest_hash':'c','selection_hash':'d','model':{'revision':'e'}}
        row={'id':'case','seed':17,'prompt_hash':'p','metrics':{'valid':True,'reward':0.0,'ari':0.0}}
        for name in ['before','after']:
            folder=root/name;folder.mkdir();(folder/'summary.json').write_text(json.dumps(summary));(folder/'predictions.jsonl').write_text(json.dumps(row)+'\n')
        return summary
    def test_identical_is_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.create(root);result=compare(root/'before',root/'after')
            self.assertEqual(result['reward_delta'],0);self.assertEqual(result['ari_delta_both_valid'],0)
    def test_changed_decoding_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);summary=self.create(root);summary['decoding']['cap']=1000
            (root/'after/summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):compare(root/'before',root/'after')
    def test_changed_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.create(root)
            path=root/'after/predictions.jsonl';row=json.loads(path.read_text());row['prompt_hash']='new';path.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):compare(root/'before',root/'after')
    def test_partial_family_metadata_uses_episode_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.create(root)
            path=root/'after/predictions.jsonl';row=json.loads(path.read_text());row['family']='topic';path.write_text(json.dumps(row)+'\n')
            result=compare(root/'before',root/'after')
            self.assertEqual(result['pairs'][0]['family'],'case')
            self.assertTrue(result['bootstrap_unit'].startswith('episode'))
    def test_conflicting_family_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);self.create(root)
            for name in ['before','after']:
                path=root/name/'predictions.jsonl';row=json.loads(path.read_text());row['family']=name;path.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):compare(root/'before',root/'after')
