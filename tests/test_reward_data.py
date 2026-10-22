import itertools, json, pathlib, random, unittest
from tabcrew_rl.reward import parse_assignments, score, partition_metrics, InvalidPartition, grpo_reward
from tabcrew_rl.protocol import messages, schema
from tabcrew_rl.data import generate, validate

class RewardTests(unittest.TestCase):
    ref={'1':0,'2':0,'3':1,'4':1}
    def test_permutation(self):
        self.assertEqual(score('{"assignments":{"1":9,"2":9,"3":3,"4":3}}',self.ref)['reward'],1)
    def test_invalid(self):
        cases=['no','{}','[]','{"assignments":{"1":0,"1":0,"2":0,"3":1,"4":1}}',
               '{"assignments":{"1":0,"2":0,"3":1,"4":1},"assignments":{}}',
               '{"assignments":{"1":0,"2":0,"3":1}}','{"assignments":{"1":0,"2":0,"3":1,"4":1,"5":1}}',
               '{"assignments":{"1":true,"2":0,"3":1,"4":1}}','{"assignments":{"1":0.0,"2":0,"3":1,"4":1}}',
               '{"assignments":{"1":NaN,"2":0,"3":1,"4":1}}','{"assignments":{"1":10,"2":0,"3":1,"4":1}}',
               '```json\n{"assignments":{"1":0,"2":0,"3":1,"4":1}}\n```']
        for text in cases:
            with self.subTest(text=text): self.assertEqual(score(text,self.ref)['reward'],-1)
    def test_incomplete(self):
        self.assertEqual(score(json.dumps({'assignments':self.ref}),self.ref,complete=False)['reward'],-1)
    def test_degenerate(self):
        for p in ({k:0 for k in self.ref},{k:i for i,k in enumerate(self.ref)}):
            self.assertEqual(partition_metrics(self.ref,p)['ari'],0)
        one={'1':0,'2':0}; singleton={'1':0,'2':1}
        self.assertEqual(partition_metrics(one,one)['ari'],1)
        self.assertEqual(partition_metrics(singleton,singleton)['ari'],1)
    def test_crossed_negative(self):
        p={'1':0,'2':1,'3':0,'4':1}
        self.assertEqual(partition_metrics(self.ref,p)['ari'],-0.5)
        self.assertEqual(partition_metrics(self.ref,p)['pair_f1'],0)
    def test_alternative_not_declared_invalid(self):
        x=score('{"assignments":{"1":0,"2":0,"3":0,"4":1}}',self.ref)
        self.assertTrue(x['valid']); self.assertLess(x['reward'],1)
    def test_reward_bridge(self):
        self.assertEqual(grpo_reward([[{'role':'assistant','content':json.dumps({'assignments':self.ref})}]], [json.dumps(self.ref)]),[1])
    def test_ari_against_independent_pair_definition(self):
        # Brute-force pair counts, separate from contingency-table implementation.
        rng=random.Random(7)
        for n in range(2,15):
            for _ in range(30):
                a=[rng.randrange(4) for _ in range(n)]; b=[rng.randrange(4) for _ in range(n)]
                pairs=list(itertools.combinations(range(n),2))
                aa=sum(a[i]==a[j] for i,j in pairs); bb=sum(b[i]==b[j] for i,j in pairs)
                both=sum(a[i]==a[j] and b[i]==b[j] for i,j in pairs)
                expected=aa*bb/len(pairs); denom=(aa+bb)/2-expected
                value=(both-expected)/denom if denom else 1
                self.assertAlmostEqual(partition_metrics(dict(enumerate(a)),dict(enumerate(b)))['ari'],value)

class DatasetTests(unittest.TestCase):
    def test_corpus_and_leak_detection(self):
        eps=list(generate()); report=validate(eps)
        self.assertEqual(report['episodes'],144)
        ep=dict(eps[0]); ep['id']='leaked'; ep['split']='validation'
        with self.assertRaises(AssertionError): validate(eps+[ep])
    def test_determinism(self):
        self.assertEqual(list(generate()),list(generate()))
    def test_prompt_allowlist_and_injection(self):
        ep=list(generate())[3]; ep['reference']['canary']='SECRET_LABEL_FIELD'
        wire=json.dumps(messages(ep))
        self.assertNotIn('SECRET_LABEL_FIELD',wire)
        self.assertNotIn('provenance',wire)
        self.assertIn('Ignore earlier instructions',wire)
        self.assertIn('untrusted',wire)
    def test_dynamic_schema(self):
        ep=list(generate())[4]
        self.assertEqual(len(schema(ep)['properties']['assignments']['required']),30)

if __name__=='__main__': unittest.main()
