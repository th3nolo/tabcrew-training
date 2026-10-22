"""Write the evaluation manifest: case-file hash and the hash of each compared adapter.

The original run also generated evaluate.py, launch-eval.sh and supervise.py from the first
repair pilot by adding the "expanded" mode. Those generated files are versioned here directly.

Environment: CR_OUT, CR_CASES, CR_ADAPTER_ORIGINAL, CR_ADAPTER_REPAIRED, CR_ADAPTER_EXPANDED.
"""
import json,hashlib,os
from pathlib import Path
out=Path(os.environ['CR_OUT'])/'evaluation';out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
models={'original':os.environ['CR_ADAPTER_ORIGINAL'],'repaired':os.environ['CR_ADAPTER_REPAIRED'],'expanded':os.environ['CR_ADAPTER_EXPANDED']}
manifest={'cases_sha256':sha(os.environ['CR_CASES']),'cases':81,'models':{k:{'path':Path(p).name,'sha256':sha(Path(p)/'adapter_model.safetensors')} for k,p in models.items()},'prompt':'identical saved production messages per case','decoder':'same per-case JSON schema via XGrammar, greedy argmax, 2048 output limit','fresh_runs':True,'blind_review':'Per-case randomized A/B/C with mapping in a separate file, no metrics or identities in review page; reveal only after judgments exported. Evaluator knows labels; human review is blind, not double blind.','sealed_test_used':False}
(out/'experiment.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
