# TabCrew training code

TabCrew groups browser tabs. The input is a session of 10 to 30 tabs (title and URL).
The output is JSON that puts every tab in one group, for example `{"assignments":{"1":0,"2":1,"3":0}}`.

This repository holds the code and recipes that trained small LoRA adapters for this task on
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)
(revision `2fc06364715b967f1860aea9cf38778875588b17`). Training used SFT and GRPO on one RTX 3080 (10 GB).
The training data is synthetic: LLM teachers wrote invented tab sessions and their groupings.

Weights and data are on Hugging Face:

| What | Link |
|---|---|
| Mixed-teacher SFT adapter (main result) | [th3nolo/tabcrew-qwen3.5-0.8b-lora-mixed-sft](https://huggingface.co/th3nolo/tabcrew-qwen3.5-0.8b-lora-mixed-sft) |
| Early 705-session adapters (SFT, then GRPO) | [th3nolo/tabcrew-qwen3.5-0.8b-lora-early-705](https://huggingface.co/th3nolo/tabcrew-qwen3.5-0.8b-lora-early-705) |
| Synthetic tab sessions | [th3nolo/tabcrew-synthetic-tab-sessions](https://huggingface.co/datasets/th3nolo/tabcrew-synthetic-tab-sessions) |

## Results

Validation set: 81 fixed sessions (30 IFM, 21 local Qwen, 30 DeepSeek), selected before any predictions.
All three adapters use the same prompt, the same greedy decoding (512 new tokens, no thinking) and the same scoring.
"Exact grouping" means the predicted groups are the same as the reference groups (group numbers may differ).

| Adapter | Valid JSON output | Exact grouping | Latency p50 (RTX 3080) |
|---|---:|---:|---:|
| Mixed-teacher SFT (this repo: `experiments/mixed-sft`) | 100% | 95.1% | 2.21 s |
| Early SFT (`experiments/early-705`) | 98.8% | 55.6% | 2.93 s |
| Early SFT + GRPO (`experiments/early-705`) | 97.5% | 54.3% | 3.99 s |

Final test (369 sessions): TBD

These numbers measure agreement with synthetic teacher labels. They are not a measure of agreement with real users.
The mixed run changes both the data and the batch size (1 to 4), so the gain is not a data-only effect.

## Hardware and software

- GPU: NVIDIA RTX 3080, 10 GB. Windows 11 host; the contract-repair run and the SGLang teacher run in WSL2 (Ubuntu).
- Python 3.11, `uv`, PyTorch 2.8.0 (CUDA 12.8), transformers 5.12.0, TRL 0.29.0, PEFT 0.18.1. All versions are pinned in `pyproject.toml` and `uv.lock`.
- LoRA rank 8, alpha 16, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`. Base weights load in bfloat16.

## Layout

| Folder | Contents |
|---|---|
| `tabcrew_rl/` | Prompt (`protocol.py`), strict JSON reward and partition metrics (`reward.py`), generated curriculum (`data.py`), model loading (`model.py`), SFT (`sft.py`), GRPO (`train.py`), evaluation (`evaluate.py`), paired comparison (`compare.py`) |
| `tests/` | Unit tests for the reward, the prompt, target serialization, the comparison, completion batching and experiment file consistency |
| `configs/` | GRPO settings for the first pilot |
| `scripts/` | Setup, model download with hash checks, bounded runs, adapter export, plots, offline output checks |
| `experiments/early-705/` | SFT then GRPO on the first 705 teacher-labelled sessions |
| `experiments/mixed-sft/` | One-epoch SFT on 3,766 sessions from three teachers, and the matched comparison |
| `experiments/contract-repair/` | Continuation SFT on 2,973 corrected labels in the production prompt format, in a network-isolated sandbox |
| `data-generation/` | The teacher collectors that wrote the synthetic sessions (code only, no data) |
| `recipes/speed/` | SGLang settings and measurements for a 4B AWQ teacher on an RTX 3080, and kernel experiments |

## Setup

Run in PowerShell from the repository root. You need `uv` and a CPython 3.11 install.

```powershell
.\scripts\bootstrap.ps1 -PythonPath C:\path\to\python3.11.exe
```

This does four things:

1. `scripts/audit_dependencies.py` checks every package in `uv.lock` on PyPI: the version exists and it is at least 72 hours old.
2. `uv sync --frozen` installs the locked versions into `work/venv-rl`.
3. `scripts/check_stack.py` writes the installed versions and GPU details to `evidence/stack-check.json`.
4. `scripts/download_model.py` downloads the pinned base-model revision to `work/models/<revision>/` and checks every file hash.

All local state goes to `work/` (ignored by git). Set `TABCREW_WORK_DIR` to use another folder.

Run the unit tests (CPU only, no model is loaded):

```powershell
.\scripts\run.ps1 -Action test
```

## Reproduce the experiments

Each experiment folder has its own README with the exact commands and inputs.

1. **First pilot on generated data.** `.\scripts\reproduce.ps1 -Name pilot1` generates the curriculum in `data/v1`
   (`python -m tabcrew_rl.data`), then runs a baseline evaluation, GRPO, SFT, GRPO after SFT, a fixed final-test
   evaluation of all four and a paired comparison. Every stage has a hard time limit. Results go to `runs/pilot1/`.
2. **Early 705 sessions** (`experiments/early-705/README.md`). Needs the early-collection splits from the dataset repository.
3. **Mixed-teacher SFT** (`experiments/mixed-sft/README.md`). Needs the mixed splits and the two early adapters for the comparison.
4. **Contract repair** (`experiments/contract-repair/README.md`). Needs Linux or WSL2 and bubblewrap. The corrected labels and the production prompt are not published, so this experiment documents the method only.

## Teachers

The synthetic sessions were written and checked by these teachers (details in `data-generation/README.md`):

- IFM API, model `K2-Horizon-375B-A23B`.
- DeepSeek API, model `deepseek-v4-flash`.
- Local `Qwen/Qwen3-4B-AWQ` served by SGLang 0.5.17 on the RTX 3080.
- Local `Qwen3-8B` (GGUF `Q4_K_M`) served by llama.cpp on the RTX 3080.
- The first 705 sessions (`experiments/early-705`) came from an earlier hosted LLM teacher. Its collector is not part of this repository.

## Limitations

- All labels come from LLM teachers. A teacher and its blind reviewer agreed on each admitted session. No human checked the labels.
- Sessions are invented. They do not come from real browsing history.
- Validation latency is GPU latency inside Python. It is not browser, CPU or server latency.
- One training seed (42) per run. No hyperparameter search.
- The data collectors cannot start a new corpus from this repo alone. They expect run-state files from the original runs (`migration.json`, `inherited-dataset.json`, `budget-origin.json`), which are not published.
- PyTorch 2.8.0 has two published issues: [CVE-2025-55554](https://osv.dev/vulnerability/PYSEC-2025-206) (integer overflow in `torch.nan_to_num`, fixed in 2.9.0) and [CVE-2025-55552](https://security-tracker.debian.org/tracker/CVE-2025-55552) (crash with `torch.rot90` plus `torch.randn_like`). This code does not call those functions. The version stays at 2.8.0 because the published adapters were trained with it.

## License

Apache-2.0 for the code in this repository (see `LICENSE` and `NOTICE`).
The base model is not included; see its own license on its model page.
