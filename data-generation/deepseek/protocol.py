"""Direct DeepSeek synthetic data collection. No local model runtime is started."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

OUT = Path(__file__).resolve().parent
# Shared helpers (normalizer, duplicate index, verifier schema) from the earlier local llama.cpp collector.
LOCAL = OUT.parent / "local-llamacpp"
sys.path.insert(0, str(LOCAL))
import generate as g
import audience

MODEL = "deepseek-v4-flash"
REVISION = "DeepSeek-V4-Flash-0731"
VERSION = "deepseek-research-groups-v5"
TERMS = "https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html"
PRICING = "https://api-docs.deepseek.com/quick_start/pricing/"


class StopCollection(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise StopCollection("Provider redirects are forbidden")


def schema_check(value, schema):
    """Validate the JSON Schema subset used by the frozen local author/reviewer."""
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "integer": type(value) is int}
    if kind and not valid.get(kind, False):
        raise ValueError("json_schema_type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("json_schema_enum")
    if kind == "object":
        props = schema["properties"]
        if not set(schema.get("required", [])).issubset(value):
            raise ValueError("json_schema_required")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ValueError("json_schema_extra_keys")
        for key, item in value.items():
            schema_check(item, props[key])
    if kind in ("array", "string"):
        suffix = "Items" if kind == "array" else "Length"
        if not schema.get("min" + suffix, 0) <= len(value) <= schema.get("max" + suffix, float("inf")):
            raise ValueError("json_schema_length")
        if kind == "array":
            for item in value:
                schema_check(item, schema["items"])
    if kind == "integer" and not schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf")):
        raise ValueError("json_schema_range")


def peak_at(now):
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def estimated_cost(usage, peak=True):
    prompt = int(usage["prompt_tokens"])
    hit = int(usage.get("prompt_cache_hit_tokens", 0))
    completion = int(usage["completion_tokens"])
    if not 0 <= hit <= prompt or completion < 0:
        raise StopCollection("Invalid provider usage metadata")
    factor = 2 if peak else 1
    return factor * ((prompt - hit) * .22 + hit * .007 + completion * .66) / 1_000_000


def append_event(path, event):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def specifications():
    return audience.specifications(g)


def export_snapshot(accepted):
    episodes = sorted(accepted.values(), key=lambda e: e["id"])
    validation = g.validate_corpus(episodes)
    folder = OUT / "snapshots" / f"n{len(episodes):05d}-{g.digest([e['id'] for e in episodes])[:10]}"
    if folder.exists():
        return str(folder)
    folder.mkdir(parents=True)
    manifest = {"generator": VERSION, "teacher": MODEL, "created_at": g.utc(), "validation": validation,
                "status": "synthetic_consistency_validated_not_human_reviewed", "splits": {},
                "output_use_permission": TERMS, "output_permission_clause": "4.2(3)",
                "source_corpus": "DeepSeek_v4_byte_identical_carryover_plus_v5_compact_verifier",
                "curriculum": audience.VERSION,
                "audience_counts": dict(Counter(ep["research_context"]["audience"] for ep in episodes)),
                "scenario_counts": dict(Counter(ep["research_context"]["scenario"] for ep in episodes))}
    for split in ("train", "validation", "final_test"):
        selected = [ep for ep in episodes if ep["split"] == split]
        file = folder / (split + ".jsonl")
        file.write_text("".join(json.dumps(ep, ensure_ascii=False) + "\n" for ep in selected), encoding="utf-8")
        manifest["splits"][split] = {"file": file.name, "count": len(selected), "sha256": g.sha(file)}
    g.atomic(folder / "manifest.json", manifest)
    return str(folder)


