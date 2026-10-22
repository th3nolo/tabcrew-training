"""Local, serial training run on a frozen snapshot of the 705-session early collection.

Stages: SFT (one epoch), validation, GRPO after SFT, validation, base-model validation.
Uses the pinned project environment and model snapshot. No teacher or provider client is run.

Paths (all optional):
  TABCREW_RUN_DIR          where dataset/, sft/, grpo/, logs and receipts are written (default: this folder)
  TABCREW_WORK_DIR         model snapshot and caches (default: <repo>/work, see tabcrew_rl/model.py)
  TABCREW_EARLY705_SOURCE  folder with train.jsonl, validation.jsonl and final_test.jsonl of the
                           early collection, as published in the dataset repository ("prepare" only)
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RUN = Path(os.environ.get("TABCREW_RUN_DIR") or ROOT).resolve()
DATA = RUN / "dataset"
GRPO_NOTE = "705-example run: after one complete SFT epoch; frozen train only; beta=0; stop on three equal-reward groups"
sys.path.insert(0, str(PROJECT))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                  PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                  PYTHONPATH=str(PROJECT))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def prepare(source):
    """Snapshot the published early collection into DATA and write the run plan.

    The original run rebuilt these episodes from the collector's accepted batches and
    checked them against the collector contract. The published dataset already holds
    the normalized episodes, so this version reads them directly. It keeps every
    other check: curriculum validation, split counts, tab limit and token budget.
    """
    from tabcrew_rl.data import validate, load_split
    from tabcrew_rl.model import model_dir

    if DATA.exists():
        raise ValueError("Snapshot already exists; do not overwrite frozen data")
    source = Path(source)
    episodes, hashes = [], {}
    for split in ("train", "validation", "final_test"):
        path = source / (split + ".jsonl")
        rows = read_rows(path)
        if any(ep["split"] != split for ep in rows):
            raise ValueError("Split label mismatch in " + path.name)
        episodes.extend(rows)
        hashes[path.name] = sha(path)
    report = validate(episodes)
    if report["splits"] != {"train": 570, "validation": 70, "final_test": 65}:
        raise ValueError("Unexpected corpus size: " + str(report))
    if any(ep["tab_count"] > 30 for ep in episodes):
        raise ValueError("Run must be redesigned to include longer examples, not silently drop them")
    snapshot = model_dir()
    if not (snapshot / "download-receipt.json").is_file():
        raise ValueError("Local pinned model receipt missing; run scripts/download_model.py")
    from transformers import AutoTokenizer
    from tabcrew_rl.protocol import messages, canonical_reference
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    token_lengths = []
    # Token-budget audit covers train/validation only; final-test content is not tokenized.
    for ep in episodes:
        if ep["split"] == "final_test":
            continue
        prefix = tok.apply_chat_template(messages(ep), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        target = canonical_reference(ep["reference"], [t["ref"] for t in ep["tabs"]])
        lengths = (len(tok.encode(prefix, add_special_tokens=False)),
                   len(tok.encode(json.dumps(target, separators=(",", ":")), add_special_tokens=False)) + 1)
        if lengths[0] > 4096 or lengths[1] > 512:
            raise ValueError("Token budget requires redesign: " + ep["id"] + " " + str(lengths))
        token_lengths.append(lengths)
    DATA.mkdir(parents=True)
    manifest = {"created_at": utc(), "status": "frozen_partial_collection_705_of_5000",
                "teacher": "hosted LLM teacher (early collection)", "source_files_sha256": hashes,
                "human_review": "Not independently semantically reviewed",
                "validation": report, "splits": {},
                "tab_count_distribution": dict(Counter(ep["tab_count"] for ep in episodes)),
                "max_prompt_tokens_train_validation": max(x[0] for x in token_lengths),
                "max_target_tokens_train_validation": max(x[1] for x in token_lengths)}
    for split in ("train", "validation", "final_test"):
        selected = sorted((ep for ep in episodes if ep["split"] == split), key=lambda ep: ep["id"])
        path = DATA / (split + ".jsonl")
        path.write_text("".join(json.dumps(ep, ensure_ascii=False, separators=(",", ":")) + "\n"
                                for ep in selected), encoding="utf-8", newline="\n")
        manifest["splits"][split] = {"file": path.name, "count": len(selected), "sha256": sha(path)}
    write(DATA / "manifest.json", manifest)
    for split in manifest["splits"]:
        load_split(DATA, split)
    # grpo-config.json and grpo_train.py are versioned next to this file. The original run
    # generated them here; this version checks that they still match the same derivation.
    config = json.loads((PROJECT / "configs" / "grpo-after-sft-pilot.json").read_text())
    config.update(max_steps=16, max_wall_seconds=900, max_tabs=30, max_completion_length=512, note=GRPO_NOTE)
    if json.loads((ROOT / "grpo-config.json").read_text(encoding="utf-8")) != config:
        raise ValueError("grpo-config.json differs from the derived configuration; review before running")
    # The original smoke trainer is preserved. This run's copy only expands prompt capacity.
    grpo_source = (PROJECT / "tabcrew_rl" / "train.py").read_text(encoding="utf-8")
    old_bound = "if len(tokenizer.encode(prompt))>2048:"
    if grpo_source.count(old_bound) != 1:
        raise ValueError("GRPO trainer prompt guard changed; review before copying")
    grpo_source = grpo_source.replace(old_bound, "if len(tokenizer.encode(prompt))>4096:")
    grpo_source = grpo_source.replace("from .", "from tabcrew_rl.")
    if (ROOT / "grpo_train.py").read_text(encoding="utf-8") != grpo_source:
        raise ValueError("grpo_train.py differs from the derived trainer copy; review before running")
    source_files = [PROJECT / "tabcrew_rl" / name for name in
                    ("model.py", "protocol.py", "reward.py", "data.py", "train.py", "evaluate.py")]
    source_files += [PROJECT / "pyproject.toml", PROJECT / "uv.lock", Path(__file__), ROOT / "grpo_train.py",
                     ROOT / "grpo-config.json"]
    write(RUN / "run-plan.json", {
        "created_at": utc(), "dataset_manifest_sha256": sha(DATA / "manifest.json"),
        "source_code_sha256": {str(p): sha(p) for p in source_files},
        "model_snapshot": str(snapshot), "model_receipt_sha256": sha(snapshot / "download-receipt.json"),
        "network_mode": "Hugging Face offline; existing local model and environment; no provider calls",
        "stages": ["SFT: 570 examples, one shuffled epoch, batch 1, LoRA r8, LR 2e-4",
                   "SFT validation: all 70 validation examples",
                   "GRPO: SFT initialization, at most 16 updates, four sampled completions per group",
                   "GRPO validation: same 70 validation examples", "Base validation: same 70 examples"],
        "final_test": "65 examples sealed; no final-test inference or optimization in this run",
        "selection": "Higher validation mean reward, then exact partition rate, then validity; prefer SFT on ties",
        "hard_stage_timeouts_seconds": {"sft": 3720, "sft-validation": 1800, "grpo": 1020,
                                        "grpo-validation": 1800, "base-validation": 1800},
        "maximum_stage_time_total_seconds": 10140,
        "provider_compute_cost_usd": 0,
        "limitations": ["Synthetic teacher references, not independent human ground truth",
                        "Teacher output-use terms for this early collection were not independently verified",
                        "Teacher labels were produced earlier by a hosted LLM; only this training is local"]})
    write(RUN / "status.json", {"state": "prepared", "updated_at": utc(), "dataset_examples": 705,
                                 "train": 570, "validation": 70, "final_test": 65})
    print(json.dumps(report), flush=True)


def sft():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    from tabcrew_rl.model import load
    from tabcrew_rl.data import load_split
    from tabcrew_rl.protocol import messages, canonical_reference, PROMPT_VERSION, TARGET_SERIALIZATION_VERSION

    out = RUN / "sft"
    out.mkdir(exist_ok=False)
    set_seed(42)
    model, tok, audit = load()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    episodes = load_split(DATA, "train")
    random.Random(42).shuffle(episodes)
    encoded = []
    for ep in episodes:
        prefix = tok.apply_chat_template(messages(ep), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok.encode(prefix, add_special_tokens=False)
        target = canonical_reference(ep["reference"], [t["ref"] for t in ep["tabs"]])
        target_ids = tok.encode(json.dumps(target, separators=(",", ":")), add_special_tokens=False) + [tok.eos_token_id]
        if len(prompt_ids) > 4096 or len(target_ids) > 512:
            raise ValueError("Token cap exceeded; no truncation permitted: " + ep["id"])
        encoded.append((ep["id"], prompt_ids, target_ids))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    write(out / "run-manifest.json", {"model": audit, "seed": 42, "learning_rate": 2e-4,
        "epochs_requested": 1, "steps_requested": len(encoded), "batch_size": 1,
        "training_episode_ids": [ep[0] for ep in encoded], "prompt_version": PROMPT_VERSION,
        "target_serialization": TARGET_SERIALIZATION_VERSION,
        "dataset_manifest_sha256": sha(DATA / "manifest.json"),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "supervision": "Completion tokens only; frozen training references; validation/final test excluded"})
    torch.cuda.reset_peak_memory_stats()
    model.train()
    start = time.perf_counter()
    losses = []
    stopped_for_time = False
    with (out / "learning-curve.jsonl").open("w", encoding="utf-8") as log:
        for step, (episode_id, prompt_ids, target_ids) in enumerate(encoded, 1):
            ids = torch.tensor([prompt_ids + target_ids], device=model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=len(target_ids) + 1).logits[:, :-1, :]
            targets = torch.tensor([target_ids], device=model.device)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SFT loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
            row = {"step": step, "total_steps": len(encoded), "episode": episode_id,
                   "loss": loss.item(), "grad_norm": float(grad), "seconds": time.perf_counter() - start}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            write(out / "progress.json", {**row, "updated_at": utc(), "state": "training"})
            if step % 100 == 0:
                checkpoint = out / ("checkpoint-" + str(step))
                model.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
            if time.perf_counter() - start >= 3600 and step < len(encoded):
                stopped_for_time = True
                break
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    delta = sum((p.detach().cpu().float() - initial[n].float()).square().sum().item()
                for n, p in model.named_parameters() if n in initial) ** 0.5
    seconds = time.perf_counter() - start
    summary = {"optimizer_steps": step, "unique_training_examples_seen": step,
               "requested_examples": len(encoded), "full_epoch_complete": step == len(encoded),
               "stopped_for_time": stopped_for_time, "nonzero_update": delta > 0, "adapter_l2_delta": delta,
               "initial_20_mean_loss": sum(losses[:20]) / len(losses[:20]),
               "last_20_mean_loss": sum(losses[-20:]) / len(losses[-20:]),
               "wall_seconds": seconds, "local_gpu_hours": seconds / 3600, "provider_compute_cost_usd": 0,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    write(out / "summary.json", summary)
    write(out / "progress.json", {"state": "completed" if not stopped_for_time else "partial",
                                   "updated_at": utc(), **summary})
    print(json.dumps(summary), flush=True)
    if stopped_for_time or delta == 0:
        raise RuntimeError("SFT incomplete or no parameter update; preserve adapter and stop pipeline")


def pipeline():
    plan = json.loads((RUN / "run-plan.json").read_text(encoding="utf-8"))
    for path, digest in plan["source_code_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Source changed since preparation: " + path)
    if sha(DATA / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    lock = RUN / "started.json"
    with lock.open("x", encoding="utf-8") as file:
        json.dump({"pid": os.getpid(), "started_at": utc()}, file)
    receipts = []
    started = time.perf_counter()
    def status(state, **extra):
        write(RUN / "status.json", {"state": state, "updated_at": utc(), "supervisor_pid": os.getpid(),
              "seconds": time.perf_counter() - started, "receipts": receipts, **extra})
    def run(stage, command):
        status("running", stage=stage)
        start = time.perf_counter()
        with (RUN / (stage + ".log")).open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, "-u", *command], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            status("running", stage=stage, child_pid=child.pid)
            try:
                code = child.wait(timeout=plan["hard_stage_timeouts_seconds"][stage])
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                code = 124
        receipt = {"stage": stage, "pid": child.pid, "exit_code": code,
                   "seconds": time.perf_counter() - start, "finished_at": utc()}
        receipts.append(receipt)
        write(RUN / (stage + ".receipt.json"), receipt)
        if code:
            raise RuntimeError(stage + " failed with exit code " + str(code))
    def evaluate(stage, adapter=None):
        command = ["-m", "tabcrew_rl.evaluate", "--data", str(DATA), "--split", "validation",
                   "--out", str(RUN / stage), "--max-tabs", "30", "--max-new-tokens", "512"]
        if adapter:
            command += ["--adapter", str(adapter)]
        run(stage, command)
    try:
        run("sft", [str(Path(__file__)), "sft"])
        evaluate("sft-validation", RUN / "sft" / "adapter")
        run("grpo", [str(ROOT / "grpo_train.py"), "--config", str(ROOT / "grpo-config.json"),
                     "--data", str(DATA), "--out", str(RUN / "grpo"),
                     "--init-adapter", str(RUN / "sft" / "adapter")])
        evaluate("grpo-validation", RUN / "grpo" / "adapter")
        evaluate("base-validation")
        comparisons = {name: json.loads((RUN / (name + "-validation") / "summary.json").read_text())
                       for name in ("sft", "grpo", "base")}
        if len({x["selection_hash"] for x in comparisons.values()}) != 1:
            raise ValueError("Validation selections differ")
        if any(x["metrics"]["episodes"] != 70 for x in comparisons.values()):
            raise ValueError("Incomplete validation")
        def key(name):
            m = comparisons[name]["metrics"]
            return (m["mean_reward_all"], m["exact_partition_rate"], m["validity_rate"], name == "sft")
        chosen = max(("sft", "grpo"), key=key)
        summary = {"completed_at": utc(), "selected_adapter": str(RUN / chosen / "adapter"),
                   "selected_stage": chosen, "validation": {n: x["metrics"] for n, x in comparisons.items()},
                   "final_test_evaluated": False, "final_test_count": 65,
                   "sft": json.loads((RUN / "sft" / "summary.json").read_text()),
                   "grpo": json.loads((RUN / "grpo" / "summary.json").read_text()),
                   "dataset_manifest_sha256": sha(DATA / "manifest.json"), "provider_compute_cost_usd": 0}
        write(RUN / "summary.json", summary)
        status("completed", selected_adapter=summary["selected_adapter"])
    except BaseException as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "sft", "pipeline"))
    parser.add_argument("--source", default=os.environ.get("TABCREW_EARLY705_SOURCE"),
                        help="Folder with the published train/validation/final_test JSONL files (prepare only)")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.source:
            parser.error("prepare needs --source or TABCREW_EARLY705_SOURCE")
        prepare(args.source)
    else:
        {"sft": sft, "pipeline": pipeline}[args.action]()
