# Data generation: teacher collectors

These folders hold the last version of each collector that wrote the synthetic tab sessions.
Only code and settings are here. Collected sessions, raw responses, request logs, billing and quota ledgers
are not included. The published sessions are in the dataset repository
[th3nolo/tabcrew-synthetic-tab-sessions](https://huggingface.co/datasets/th3nolo/tabcrew-synthetic-tab-sessions).

## How a session is made

All collectors follow the same steps:

1. **Specification.** A fixed curriculum (`audience.py`) gives each session an ID, split, language
   (`en`, `es` or mixed), tab count (10 to 30), 2 to 4 research tasks with niches, and URL templates
   for realistic platform pages (`url_templates.py`, `source_catalog.py`). Splits are fixed by curriculum
   family, so a family never appears in two splits.
2. **Author.** The teacher writes one short task description per group and one title per tab, as JSON
   that must match a strict schema. The code builds the URLs and the reference grouping.
3. **Blind review.** A second request shows the teacher only the user's visible tasks, the titles and the
   URLs, without the reference. It must assign every tab to a task.
4. **Admission.** A session is accepted only if the blind assignment equals the reference exactly, no tab
   is marked ambiguous, the title language matches the requested language, and the session is not an exact
   or near duplicate (title trigram Jaccard 0.8 or more, or the same canonical page in another split).

Accepted sessions keep their teacher, request hashes and contract hash in a `provenance` field.
Agreement between author and blind reviewer is synthetic consistency, not human validation.

## Collectors

| Folder | Teacher | Runtime | Notes |
|---|---|---|---|
| `local-sglang/` | `Qwen/Qwen3-4B-AWQ` (revision `74d4bd2b...`) | SGLang 0.5.17 in WSL2 on the RTX 3080 | 4 workers, 16,384-token KV pool; one targeted title repair (at most 8 fields) before the blind review; host RAM, commit, GPU memory and temperature guards (`settings.json`). Serving settings: `recipes/speed/README.md`. |
| `ifm/` | IFM API, `K2-Horizon-375B-A23B` | `https://api.ifm.ai/v1/chat/completions` | Up to 3 sessions (40 tabs) per author request, one blind review per session. One new request per second, no fixed concurrency cap. Honors `Retry-After` on HTTP 429. Fixed lifetime token cap. Terms notes: `ifm/POLICY.md`. |
| `deepseek/` | DeepSeek API, `deepseek-v4-flash` | `https://api.deepseek.com/chat/completions` | 4 workers, compact blind verifier (task index per tab, observed language, ambiguous refs), cumulative spending cap (at most 8 USD). Terms notes: `deepseek/TERMS.md`. |
| `local-llamacpp/` | `Qwen3-8B` GGUF `Q4_K_M` (`Qwen/Qwen3-8B-GGUF`, revision `7c41481f...`) | llama.cpp `b10809`, Windows CUDA 12.4 build | Earlier local collector. `acquire.py` downloads the pinned model and runtime and checks SHA-256. The DeepSeek collector imports its normalizer, duplicate index and verifier schema from `generate.py`. |

## Credentials

The API collectors read the key from an environment variable only:

- IFM: `IFM_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`

Each collector reads the key once, removes it from its own environment, and never writes it to arguments,
logs or reports. `start.ps1` refuses to start if the variable is not set and clears it when it exits.

## Local paths

| Variable | Used by | Meaning |
|---|---|---|
| `TABCREW_WORK_DIR` | all `start.ps1` | uv virtual environment and cache (default `<repo>/work`) |
| `TABCREW_SGLANG_RUNTIME` | `local-sglang` | Linux path of the SGLang runtime folder in WSL (`.venv`, `uv.lock`, model folder from `assets.json`) |
| `TABCREW_SGLANG_LOCK` | `local-sglang` | Windows path of a reviewed copy of that runtime's `uv.lock`; its hash is pinned in the corpus contract |
| `TABCREW_WSL_DISTRO` | `local-sglang` | WSL distribution (default `Ubuntu`) |
| `TABCREW_LLAMACPP_WORK` | `local-llamacpp` | Downloads and unpacked llama.cpp runtime (default `<repo>/work/local-generation`) |

The collectors import `tabcrew_rl` from the repository root (two folders up).

## Status of this code

These collectors ran in place, as a chain of revisions. Each revision resumed the previous one: it copied the
accepted sessions byte for byte, recorded them in `migration.json` / `inherited-dataset.json`, and checks
those records at start. Those run-state files are not published, so the code here cannot resume the original
corpus and does not have a "start an empty corpus" mode. It is published to show exactly how sessions were
written, checked and admitted. Starting a new corpus needs a small change: an empty migration record.

The collector unit tests are not included. They read accepted sessions from the earlier revision folders.

## Differences from the code that ran

- Local folder paths became the variables above. The repository root replaces the original project path.
- The PowerShell start scripts no longer read an encrypted key file. They require the environment variable.
- A local game-process check was removed from the local start script.
- One snapshot metadata value in `local-llamacpp/generate.py` that named an earlier teacher was made generic.
- `local-llamacpp/generate.py` reads its theme catalog from `themes.py`. Before, it read the same list from a module that is not part of this repository.
