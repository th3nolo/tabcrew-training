"""Frozen research-workflow curriculum; no inference or network at import time."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
VERSION = "entrepreneur-research-v2"

# Domains, including secondary tasks, never cross splits. Workflows and source
# formats intentionally recur: the transfer question is new topics, not new labels.
DOMAINS = {
    "train": [
        "specialty coffee subscriptions", "independent skincare brands", "remote coworking memberships",
        "boutique fitness studios", "sustainable packaging suppliers", "pet enrichment products",
        "language tutoring marketplaces", "small hotel booking tools", "artisan food delivery",
        "local home cleaning services", "handmade jewelry businesses", "digital bookkeeping services",
        "outdoor equipment rentals", "wedding photography studios", "B2B email software",
        "online cooking courses", "furniture refurbishment shops", "indie game marketing",
        "cycling accessories retailers", "children educational toys", "plant nursery ecommerce",
        "podcast production agencies", "cybersecurity awareness courses", "waste reduction apps",
        "ceramics creator businesses", "independent music promotion", "mobile car detailing",
        "translation service agencies", "virtual event production", "architectural visualization studios",
        "print on demand stationery", "reusable home storage products",
        "urban wildlife photography", "community choir projects", "amateur astronomy observing",
        "weekend hiking plans", "home bread baking", "restoring old furniture",
        "board game club events", "balcony vegetable gardening",
    ],
    "validation": [
        "ocean conservation tourism", "assistive reading devices", "specialty tea wholesalers",
        "theatrical costume workshops", "amateur radio antenna projects",
    ],
    "final_test": [
        "repairable travel luggage", "rural water monitoring services", "museum audio tour studios",
        "adaptive sports equipment", "historical map restoration",
    ],
}

WORKFLOWS = [
    ("market_research", "compare customer problems, competing offers and demand evidence"),
    ("people_research", "research fictional founders, experts and potential collaborators through public professional material"),
    ("creator_research", "shortlist fictional creators using audience fit, videos, posts and professional profiles"),
    ("content_research", "prepare an evidence-based article or video with complementary sources and conflicting claims"),
    ("freelancer_research", "compare fictional freelancers or agencies through portfolios, profiles, case studies and briefs"),
    ("client_research", "understand a prospective fictional client's market, positioning and project needs"),
    ("product_research", "compare products or services using independent reviews, demos and company material"),
    ("trend_research", "investigate an emerging customer trend across articles, discussions, research and social content"),
]
SOURCES = [
    ("article", "an editorial article or newsletter"),
    ("youtube", "a YouTube video, interview or channel page"),
    ("instagram", "an Instagram professional profile, post or reel"),
    ("linkedin", "a LinkedIn professional profile or company post"),
    ("research", "a research report, study abstract or industry analysis"),
    ("community", "a public community discussion or question"),
    ("company", "a company product page, demo or case study"),
    ("portfolio", "a freelancer portfolio or agency project case study"),
]

FOCUS_PAIRS = [
    ("end-customer acquisition and purchase motivations", "supplier logistics and inventory operations"),
    ("professional people and recruiting specialists", "environmental materials and product lifecycle"),
    ("creator partnerships and audience fit", "technical software integrations and data workflows"),
    ("premium product positioning and brand identity", "staff education and workplace training"),
    ("consumer-facing educational content", "wholesale business procurement and distribution"),
]


def specifications(g):
    catalog_hash = g.digest(DOMAINS)
    ordered = [(split, index, domain) for split, domains in DOMAINS.items()
               for index, domain in enumerate(domains)]
    result = []
    for turn in range(100):
        for ordinal, (split, index, domain) in enumerate(ordered):
            variant = (turn + ordinal * 7) % 100
            identifier = f"research-{split}-{index:02d}-{variant:03d}"
            seed = int(g.digest([VERSION, identifier])[:8], 16) % (2**31 - 1)
            rng = random.Random(seed)
            scenario = "related_objectives" if (variant // 10 + variant % 10) % 5 == 0 else "distinct_investigations"
            core = index < (32 if split == "train" else 4)
            pool = DOMAINS[split][:(32 if split == "train" else 4)] if core else DOMAINS[split][(32 if split == "train" else 4):]
            if len(pool) < 4:
                scenario = "related_objectives"
            count = [10, 12, 18, 24, 30][(variant // 5 + index) % 5]
            groups = 2 if scenario == "related_objectives" else rng.choice([2, 3, 4])
            sizes = [2] * groups
            for _ in range(count - sum(sizes)):
                sizes[rng.randrange(groups)] += 1
            domains = [domain] * groups if scenario == "related_objectives" or len(pool) < groups else rng.sample(pool, groups)
            domains[0] = domain
            if scenario == "distinct_investigations" and len(set(domains)) != groups:
                domains = [domain] + rng.sample([d for d in pool if d != domain], groups - 1) if len(pool) >= groups else [domain] * groups
                if len(pool) < groups:
                    scenario = "related_objectives"
            workflows = [WORKFLOWS[(variant + index + group * 3) % len(WORKFLOWS)] for group in range(groups)]
            focus_pair = FOCUS_PAIRS[(variant + index) % len(FOCUS_PAIRS)]
            source_order = list(SOURCES)
            rng.shuffle(source_order)
            # Every group shares the first two sources; no platform identifies a label.
            sources = [[source_order[j % len(source_order)][0] for j in range(size)] for size in sizes]
            result.append({"id": identifier, "family": f"research-{split.replace('_', '-')}-{index:02d}",
                "split": split, "theme": domain, "variant": variant, "seed": seed,
                "locale": "en" if variant % 10 < 5 else "es" if variant % 10 < 9 else "mixed",
                "tab_count": count, "group_sizes": sizes, "task_topics": domains,
                "workflow_ids": [w[0] for w in workflows], "workflow_instructions": [w[1] for w in workflows],
                "sources_by_group": sources, "scenario": scenario,
                "focus_by_group": [focus_pair[i % 2] if scenario == "related_objectives" else "a concrete goal inside this domain" for i in range(groups)],
                "audience": "entrepreneur_freelancer" if core else "broader_research",
                "catalog_source_sha256": catalog_hash})
    assert len(result) == 5000 and len({s["id"] for s in result}) == 5000
    return result


def author_messages(g, spec, attempt):
    language = {
        "en": "Write every task and title in English.",
        "es": "Escribe todas las tareas y todos los títulos en español. Las claves JSON siguen en inglés.",
        "mixed": "Write half the titles in each group in Spanish and half in English; use natural complete phrasing.",
    }[spec["locale"]]
    sources = dict(SOURCES)
    assignments = {f"g{i}": {"domain": domain, "objective": spec["workflow_instructions"][i],
        "exclusive_research_focus": spec["focus_by_group"][i],
        "ordered_titles": [{"source":sources[s], "language": ("es" if (i+j) % 2 else "en") if spec["locale"] == "mixed" else spec["locale"]}
            for j, s in enumerate(spec["sources_by_group"][i])]}
        for i, domain in enumerate(spec["task_topics"])}
    relation = ("These investigations share a broad industry but have DIFFERENT concrete objectives. Make the objective visible "
        "in each title; do not rely on hidden context. Each group has an EXCLUSIVE focus: do not discuss the other group's focus. "
        "Use different concrete entities, audiences or operational subproblems within those foci. Do not merge distinct decisions just because they share keywords."
        if spec["scenario"] == "related_objectives" else
        "These are separate investigations. Keep each objective coherent across all its sources.")
    prompt = (
        f"Create a synthetic browser session for {'an entrepreneur or freelancer' if spec['audience'] == 'entrepreneur_freelancer' else 'a curious independent researcher'}. "
        f"Group specifications: {json.dumps(assignments, ensure_ascii=False)}. {relation} "
        "For each group write one concrete research objective in 8-16 words and exactly one title for each ordered source. "
        "Match the title to its source: profiles describe a fictional person's professional specialty; videos concern an interview, "
        "demo or analysis; studies have a specific finding; portfolios describe relevant work. All sources in a group must help the same objective. "
        "Never group by platform, file format or research stage. The same platform appears in multiple unrelated investigations. "
        "Use varied realistic wording, questions, evidence and comparisons. Invent project circumstances, names and organizations. "
        "Do not repeat the whole task sentence or an identical template in every title; use meaningful paraphrases. "
        "Concrete entity names and objective-specific words may recur when needed for unambiguous grouping. "
        "Do not write source/platform suffixes: the local formatter attaches them. No numbered templates, group IDs, generic Home pages, "
        "personal contact details, actual profile claims or implied scraped content. Keep titles below 150 characters and task sentences below 130 characters. "
        "Follow the assigned language of EACH ordered title, including the mixed-language case. No hidden page content is available to the grouping model. "
        f"Variation seed {spec['seed']}/{attempt}. {language}")
    return [{"role": "system", "content": "Write invented browser-tab research examples. No browsing or tools. Return only the requested JSON. " + language},
            {"role": "user", "content": prompt}]


def normalize(g, authored, spec, receipt):
    episode = g.normalize(authored, spec, receipt)
    title_sources = {title.strip(): spec["sources_by_group"][i][j]
        for i in range(len(spec["group_sizes"])) for j, title in enumerate(authored["groups"][f"g{i}"]["titles"])}
    for tab in episode["tabs"]:
        source = title_sources[tab["title"]]
        # Reserved .example hosts convey format without visiting actual platforms.
        # Each source spans groups; neither paths nor hostnames encode membership.
        opaque = g.digest([spec["id"], tab["ref"]])[:12]
        tab["url"] = f"https://{spec['family']}-{source}.example/p/{opaque}"
        if source in ("youtube", "instagram", "linkedin"):
            tab["title"] += " · " + {"youtube": "YouTube", "instagram": "Instagram", "linkedin": "LinkedIn"}[source]
        if len(tab["title"]) > 180:
            raise ValueError("formatted_title_length")
    episode["research_context"] = {"audience": spec["audience"], "scenario": spec["scenario"],
        "workflows": spec["workflow_ids"], "sources": dict(Counter(title_sources.values())), "curriculum": VERSION}
    episode["provenance"]["synthetic_url_generator"] = "research_source_hosts_opaque_paths_v1"
    return episode


def verify_budget_origin(g):
    origin = json.loads((HERE / "budget-origin.json").read_text(encoding="utf-8-sig"))
    for prior in [origin] + origin.get("ancestors", []):
        old = HERE.parent / prior["source_corpus"]
        if not (old / "PAUSE").exists() or g.read_json(old / "status.json")["state"] == "running":
            raise RuntimeError("Previous corpus must remain paused to preserve the shared budget")
        if g.sha(old / "billing.jsonl") != prior["billing_prefix_sha256"]:
            raise RuntimeError("Previous corpus billing changed; reconcile shared budget before resuming")
    with (HERE / "billing.jsonl").open("rb") as stream:
        prefix = stream.read(origin["billing_prefix_bytes"])
    if hashlib.sha256(prefix).hexdigest() != origin["billing_prefix_sha256"]:
        raise RuntimeError("Cumulative budget history is missing or changed")
