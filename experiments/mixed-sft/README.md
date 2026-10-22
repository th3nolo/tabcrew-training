# Mixed-teacher SFT

One new LoRA adapter, trained from the base model on sessions from three teachers.
It is compared with the two early adapters (SFT and SFT+GRPO from `experiments/early-705`)
on the same 81 validation sessions.

## Data

| Teacher | Sessions |
|---|---:|
| IFM (`K2-Horizon-375B-A23B`) | 593 |
| Local Qwen (`Qwen3-4B-AWQ`, SGLang) | 278 |
| DeepSeek (`deepseek-v4-flash`) | 2,895 |
| Total | 3,766 |

Split: 3,030 train, 367 validation, 369 final test. The teachers' own splits are kept.
`prepare.py` removed no sessions: the duplicate checks found no exact or near-duplicate session,
no shared page across splits and no curriculum family in two splits.
The longest train/validation prompt is 2,512 tokens. Nothing is truncated.

The 81-session evaluation set has 30 IFM, all 21 local Qwen and 30 DeepSeek validation sessions.
Selection goes round-robin over language, then tab count and research niche, before any predictions.
The final test stays sealed.

## Training

One epoch, batch 4, 758 optimizer updates, LoRA r8/alpha16, learning rate 2e-4, seed 42, gradient clip 1.
The loss is the mean completion-token loss of each session, then the mean over the batch. Prompt and padding
tokens have no loss (`batching.py`). Before training, `run.py preflight` runs the longest sessions as one
padded batch and one by one and stops if the two losses differ (tolerance 1.5% relative or 0.02 absolute).
It picks the largest batch (4, 2 or 1) that stays under 6.5 GiB of CUDA memory. In the original run batch 4
passed with 5.46 GiB peak. Training took 4,732 s on the RTX 3080.

Training saves a checkpoint (adapter, optimizer, random states, position) every 100 updates. A file named
`PAUSE` in the run folder, low host memory or 180 minutes of training in one launch make it save and exit
with code 75. Run `start.ps1` again to resume.

## Files

- `common.py`: paths, hashing and atomic JSON writes. All paths can be changed with environment variables (below).
- `prepare.py`: combines the three teacher snapshots, keeps their splits, removes duplicates, selects the 81 evaluation sessions.
- `run.py`: `preflight`, `train`, `evaluate --name new|old-sft|old-grpo` and `pipeline`.
- `batching.py`: padded completion-only batches and the loss.
- `report.py`: writes `comparison.json`, `comparison.md` and `comparison.html` with slices by teacher, niche, language and tab count, and paired family-block bootstrap intervals.
- `start.ps1`: checks free RAM (4 GiB) and GPU memory (6.5 GiB), then starts or resumes `run.py pipeline`.
- `run-plan.original.json`: the plan of the original run. Paths are relative. Hashes are of the original files; see its `_note`.

## Inputs and paths

| Variable | Default | Meaning |
|---|---|---|
| `TABCREW_RUN_DIR` | this folder | Run folder: `dataset/`, `sft/`, evaluations, logs, receipts |
| `TABCREW_EARLY705_RUN_DIR` | `experiments/early-705` | Finished early run: `dataset/`, `sft/adapter`, `grpo/adapter` |
| `TABCREW_DEDUP_COLLECTOR_DIR` | `data-generation/ifm` | Collector whose `core.py` gives the duplicate index |
| `TABCREW_WORK_DIR` | `work` | Model snapshot, virtual environment, caches |

`prepare.py` reads `dataset/raw/<teacher>/` for `ifm`, `local_qwen` and `deepseek`. Each folder holds
`train.jsonl`, `validation.jsonl`, `final_test.jsonl` and a `manifest.json` with
`{"splits": {"train": {"file": ..., "count": ..., "sha256": ...}, ...}}`.
It also reads the early run's `dataset/` to keep old train/validation sessions out of the new held-out sets.

If you use the combined splits from the dataset repository, skip `prepare.py`: put `train.jsonl`,
`validation.jsonl`, `final_test.jsonl` and a `manifest.json` in the same format into `dataset/`, and the
81 evaluation sessions into `dataset/evaluation/` (`validation.jsonl` and `manifest.json`).

## Run it

From the repository root, after `scripts/bootstrap.ps1` and the early-705 run:

```powershell
uv run --frozen python experiments/mixed-sft/prepare.py
uv run --frozen python experiments/mixed-sft/run.py preflight
.\experiments\mixed-sft\start.ps1
```

`run.py` uses Windows file locks (`msvcrt`). `pipeline` checks the hashes in `run-plan.json` before each
launch and stops if code, data or the early adapters changed.
