import copy,random,unittest
from tabcrew_rl.protocol import canonical_reference
from tabcrew_rl.reward import partition_metrics

class TargetSerializationTests(unittest.TestCase):
    def test_equivalent_partition_and_key_permutations_have_one_target(self):
        a={'assignments':{'10':9,'2':4,'1':9}}
        b={'assignments':{'1':1,'10':1,'2':8}}
        self.assertEqual(canonical_reference(a,[1,2,10]),canonical_reference(b,[1,2,10]))
        self.assertEqual(list(canonical_reference(a,[1,2,10])['assignments']),['1','2','10'])
    def test_partition_preserved_and_source_unchanged(self):
        rng=random.Random(78)
        for size in range(2,31):
            refs=list(range(1,size+1));rng.shuffle(refs)
            source={'assignments':{str(r):rng.randrange(10) for r in refs}};old=copy.deepcopy(source)
            result=canonical_reference(source,refs)
            self.assertEqual(source,old)
            self.assertEqual(partition_metrics(source['assignments'],result['assignments'])['ari'],1)
    def test_invalid_reference_is_not_repaired(self):
        with self.assertRaises(ValueError):canonical_reference({'assignments':{'1':True}},[1])
        with self.assertRaises(ValueError):canonical_reference({'assignments':{'1':0}},[1,2])
