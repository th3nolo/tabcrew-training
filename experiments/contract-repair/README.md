# Contract repair: continuation SFT in the production prompt format

The mixed-SFT adapter was trained with the research prompt in `tabcrew_rl/protocol.py`.
The production service uses its own system prompt and output JSON schema. This run continues the
mixed-SFT adapter for one epoch on 2,973 corrected labels written in that production format.

The production system prompt, the schema and the 2,973 labels are not published. The code reads them from local data files:
`train.jsonl` rows have `id`, `messages` (the chat messages) and `target` (the expected JSON object).
The evaluation cases file has, per case, `id`, `messages`, `schema`, `messages_sha256` and `schema_sha256`.

## Recipe

`training-plan.json` holds the settings: one epoch, batch 4, learning rate 2e-4, seed 42, gradient clip 1,
a checkpoint every 64 updates, no RL. The completion-only loss is the same as in `experiments/mixed-sft`
(`batching.py` is the same file). Prompt plus target may not exceed 8,192 tokens and the target may not exceed
2,048 tokens; nothing is truncated. The last checkpoint is number 744 (2,973 sessions / batch 4, rounded up).

`training-plan.json` pins the SHA-256 of `train.jsonl`, of the starting adapter, of the base weights and of
`train.py` and `batching.py`. `train.py` stops if any of them differ. The `train.py` hash was updated for
this repository because two local-path checks were rewritten (see below).

## Sandbox

`launch-train.sh` and `launch-eval.sh` run Python inside bubblewrap in a new, empty network namespace (no network), with no Linux
capabilities, a cleared environment and read-only inputs. `train.py` and `evaluate.py` check this at start:
not root, only the loopback route, empty effective capabilities, no `/mnt` and no `.ssh` folder under `/home`.
`supervise.py` holds an exclusive GPU job lock, stops the process tree above 4 GiB RSS or 120 minutes, and
writes a receipt.

Software: Linux or WSL2 with `/dev/dxg`, bubblewrap, a Python 3.12 environment with torch 2.11.0,
transformers 5.12.1 and xgrammar 0.2.1 (the original run used the SGLang 0.5.17 environment from
`recipes/speed`), plus peft 0.18.1 and accelerate 1.12.0 from `uv.lock` (pure Python packages, bound read-only).

## Environment variables

| Variable | Meaning |
|---|---|
| `CR_VENV` | Linux virtual environment with torch, transformers, xgrammar |
| `CR_PYTHON_BASE` | Base Python install that `CR_VENV` links to |
| `CR_EXTRA_PACKAGES` | Folder with `peft`, `accelerate` and their `*.dist-info` folders |
| `CR_BASE_MODEL` | Pinned Qwen3.5-0.8B snapshot (`work/models/<revision>`) |
| `CR_ADAPTER` | Mixed-SFT adapter to continue |
| `CR_TRAIN_JSONL` | Corrected training labels |
| `CR_CASES` | 81 evaluation cases |
| `CR_ADAPTER_ORIGINAL`, `CR_ADAPTER_REPAIRED`, `CR_ADAPTER_EXPANDED` | Adapters compared by `launch-eval.sh` |
| `CR_OUT` | Output root: `training/`, `evaluation/`, logs, lock, receipts |

## Run it

```bash
python3 supervise.py train                 # training, resumes from the last checkpoint
python3 export-curve.py &                  # optional: loss history from saved checkpoints
python3 prepare-evaluation.py              # evaluation manifest (case and adapter hashes)
for m in original repaired expanded; do python3 supervise.py "$m"; done
```

`evaluate.py` decodes greedily with the per-case JSON schema enforced by XGrammar, up to 2,048 tokens,
and records latency, time to first token and peak GPU memory. `repaired` is an earlier 128-label pilot
(checkpoint 32); it is not published.

## Changes from the original run

- Local paths in the launch scripts became the `CR_*` variables above.
- The sandbox checks for `/mnt/<drive>` and one user's `.ssh` folder became checks for any `/mnt` and any `/home/*/.ssh`.
- `prepare-evaluation.py` now only writes the manifest. The original also generated `evaluate.py`,
  `launch-eval.sh` and `supervise.py` from the 128-label pilot by adding the `expanded` mode; those files are here directly.
