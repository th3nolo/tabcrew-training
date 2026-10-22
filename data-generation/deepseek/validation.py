"""The frozen normalizer with a narrowly corrected terminal-punctuation screen."""
from collections import Counter
import random
import re
from types import SimpleNamespace
import generate as g
import audience


def check_text_quality(value):
    text = value.strip()
    # Closing quotes/brackets and percentages are valid title endings. Keep the
    # prior cutoff and broken/control-character checks; do not infer truncation
    # merely from an apostrophe or percent sign.
    if (len(text) >= 178 or "\ufffd" in text or any(ord(c) < 32 or ord(c) == 127 for c in text)
            or not re.search(r'''[\w.!?\u2026)\]}'"\u2019\u201d\u00bb\u203a%]$''', text)):
        raise ValueError("truncated_or_malformed_text")


def base_normalize(authored, spec, author_receipt):
    if not isinstance(authored, dict) or set(authored) != {"groups"}:
        raise ValueError("author_root_schema")
    if not isinstance(authored["groups"], dict) or set(authored["groups"]) != {f"g{i}" for i in range(len(spec["group_sizes"]))}:
        raise ValueError("author_groups")
    titles, seen = [], set()
    for i, size in enumerate(spec["group_sizes"]):
        group = authored["groups"][f"g{i}"]
        if not isinstance(group, dict) or set(group) != {"task", "titles"} or not isinstance(group["task"], str) or not 10 <= len(group["task"]) <= 180:
            raise ValueError("author_group_schema")
        check_text_quality(group["task"])
        if not isinstance(group["titles"], list) or len(group["titles"]) != size:
            raise ValueError("author_title_count")
        for title in group["titles"]:
            if not isinstance(title, str) or not 15 <= len(title.strip()) <= 180:
                raise ValueError("title_length")
            title = title.strip()
            check_text_quality(title)
            key = " ".join(re.findall(r"\w+", title.casefold()))
            if len(key.split()) < 4:
                raise ValueError("uninformative_title")
            if key in seen:
                raise ValueError("duplicate_title")
            if re.search(r"\b(?:group|grupo)\s*[0-9a-d]\b|\bg[0-3]\b", title, re.I):
                raise ValueError("group_label_in_title")
            seen.add(key)
            titles.append((i, title))
    random.Random(spec["seed"]).shuffle(titles)
    tabs, assignments, within_group = [], {}, Counter()
    hosts = ["notes", "guides", "community"][:min(3, min(spec["group_sizes"]))]
    for ref, (group, title) in enumerate(titles, 1):
        host = hosts[(within_group[group] + group) % len(hosts)]
        within_group[group] += 1
        opaque = g.digest([spec["id"], ref])[:12]
        tabs.append({"ref":ref, "title":title, "url":f"https://{spec['family']}-{host}.example/p/{opaque}"})
        assignments[str(ref)] = group
    reference = g.canonical_reference({"assignments":assignments}, range(1,len(tabs)+1))
    tasks = [authored["groups"][f"g{i}"]["task"] for i in range(len(spec["group_sizes"]))]
    g.check_language([title for _, title in titles], spec["locale"])
    intent = (("Tengo estas tareas independientes abiertas: " if spec["locale"] == "es" else "I have these independent tasks open: ")
              + "; ".join(t.rstrip(". ") for t in tasks) + ".")
    return {"schema_version":1, "id":spec["id"], "family":spec["family"], "split":spec["split"],
        "locale":spec["locale"], "tab_count":len(tabs), "intent":intent, "tabs":tabs, "reference":reference,
        "provenance":{"author_request_sha256":author_receipt["request_sha256"], "spec_sha256":g.digest(spec),
                      "private_data":False, "intent_generator":"teacher_task_descriptions_joined_v2"}}


def normalize(authored, spec, receipt):
    return audience.normalize(SimpleNamespace(normalize=base_normalize, digest=g.digest), authored, spec, receipt)
