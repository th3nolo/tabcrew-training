# Early 705-session run: SFT, then GRPO

This was the first training run on teacher-labelled sessions. The 705 sessions came from an earlier
hosted LLM teacher. The split is 570 train, 70 validation and 65 final test. The final test was not used.

## Stages

`run.py pipeline` runs these stages in order. Each stage is a child process with a hard time limit.

| Stage | What it does | Time limit |
|---|---|---:|
| `sft` | One shuffled epoch over 570 sessions, batch 1, LoRA r8/alpha16, learning rate 2e-4, seed 42. Loss on completion tokens only. | 3,720 s |
| `sft-validation` | Greedy evaluation of the SFT adapter on all 70 validation sessions | 1,800 s |
| `grpo` | `grpo_train.py` with `grpo-config.json`: starts from the SFT adapter, at most 16 updates, 4 samples per prompt, beta 0, learning rate 2e-5 | 1,020 s |
| `grpo-validation` | Same evaluation for the GRPO adapter | 1,800 s |
| `base-validation` | Same evaluation for the base model without an adapter | 1,800 s |

The original run took 1,036 s for SFT (570 updates) and 262 s for GRPO (16 updates).
On the 70 validation sessions: base model 58.6% valid JSON, 0% exact; SFT 92.9% valid, 4.3% exact,
mean reward 0.529; GRPO 92.9% valid, 4.3% exact, mean reward 0.532. The pipeline selected GRPO.
The repository README reports both adapters on the later 81-session mixed validation set.

## Files

- `run.py`: `prepare`, `sft` and `pipeline` actions.
- `grpo_train.py`: a copy of `tabcrew_rl/train.py` with the prompt limit raised from 2,048 to 4,096 tokens and absolute imports. `prepare` checks that this copy still matches.
- `grpo-config.json`: `configs/grpo-after-sft-pilot.json` with `max_steps=16`, `max_wall_seconds=900`, `max_tabs=30`, `max_completion_length=512`. `prepare` checks this too.
- `run-plan.original.json`: the plan of the original run (settings, stage limits, file hashes). Paths are relative. The hashes are of the original files; see its `_note`.

## Run it

From the repository root, after `scripts/bootstrap.ps1`:

```powershell
$env:TABCREW_EARLY705_SOURCE = 'C:\data\early-705'   # folder with train.jsonl, validation.jsonl, final_test.jsonl
uv run --frozen python experiments/early-705/run.py prepare
uv run --frozen python experiments/early-705/run.py pipeline
```

`prepare` validates the sessions (`tabcrew_rl.data.validate`: unique IDs, refs 1..n, valid reference JSON,
no family, title set or domain shared across splits), checks the counts 570/70/65, checks that no session
has more than 30 tabs, tokenizes train and validation and stops if a prompt is over 4,096 tokens or a target
is over 512 tokens. It writes `dataset/`, `run-plan.json` and `status.json`. `pipeline` refuses to start if
any source file or the dataset manifest changed after `prepare`.

Outputs go to this folder by default (`dataset/`, `sft/`, `grpo/`, `*-validation/`, logs, receipts).
Set `TABCREW_RUN_DIR` to write them somewhere else. The mixed-SFT experiment reads the two adapters from
`sft/adapter` and `grpo/adapter` of this run folder.

## Difference from the original run

The original `prepare` rebuilt the sessions from the collector's saved batches and checked them against
that collector's contract. That collector is not in this repository. This version reads the published
splits directly and keeps all other checks. Training and evaluation code are unchanged.
