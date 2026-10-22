"""Four bounded candidate workers, cumulative billing, and durable stage checkpoints."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from contextlib import contextmanager
import math
import threading
import protocol as p
from protocol import (OUT, LOCAL, MODEL, REVISION, VERSION, TERMS, PRICING, g, audience,
                      Counter, datetime, timezone, json, os, time, uuid, argparse,
                      urllib, StopCollection, NoRedirect, schema_check, peak_at,
                      estimated_cost, append_event, specifications, export_snapshot)
import validation
import compact_verifier as verifier


def durable_json(path, value):
    """One writer per path; fsync before atomic replacement on the same volume."""
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows readers can briefly deny delete sharing on the old file. Keep the
    # durable temporary file intact and retry replacement, without changing ACLs.
    for attempt in range(9):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(.02 * 2**attempt, .25))


@contextmanager
def collector_lock(root=OUT):
    import msvcrt
    with (root / "collector.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        yield


class Ledger:
    def __init__(self, root, budget):
        if not math.isfinite(budget) or not 0 < budget <= 8:
            raise StopCollection("invalid_budget")
        self.root, self.budget = root, budget
        self.log = root / "billing.jsonl"
        self.lock = threading.RLock()
        self.pending, self.settled = {}, set()
        self.estimated_usd = self.ceiling_usd = 0.0
        self.usage, self.calls = Counter(), 0
        if self.log.exists():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                self.apply(json.loads(line))
        if self.pending:
            raise StopCollection("previous_request_billing_unresolved_inspect_before_resume")

    def apply(self, event):
        ident = event["id"]
        if event["type"] == "reserve":
            amount = event["reserved_usd"]
            if ident in self.pending or ident in self.settled or not math.isfinite(amount) or amount <= 0:
                raise StopCollection("invalid_billing_reservation")
            self.pending[ident] = amount
        elif event["type"] == "settle":
            if ident not in self.pending or ident in self.settled:
                raise StopCollection("invalid_or_duplicate_billing_settlement")
            estimate, ceiling = event["estimated_usd"], event["ceiling_usd"]
            if not all(math.isfinite(x) and x >= 0 for x in (estimate, ceiling)) or estimate > ceiling:
                raise StopCollection("invalid_billing_cost")
            self.pending.pop(ident)
            self.settled.add(ident)
            self.estimated_usd += estimate
            self.ceiling_usd += ceiling
            self.usage.update(event["usage"])
            self.calls += 1
        else:
            raise StopCollection("unknown_billing_event")

    def reserve(self, amount, ident, stop_check, **metadata):
        with self.lock:
            stop_check()
            if not math.isfinite(amount) or amount <= 0:
                raise StopCollection("invalid_reservation")
            if self.ceiling_usd + sum(self.pending.values()) + amount > self.budget:
                raise StopCollection("budget_cap")
            event = {"type": "reserve", "id": ident, "at": g.utc(), "reserved_usd": amount, **metadata}
            append_event(self.log, event)
            self.apply(event)

    def settle(self, ident, usage):
        if any(type(usage[k]) is not int or usage[k] < 0 for k in
               ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens")):
            raise StopCollection("invalid_provider_usage_billing_unresolved")
        ceiling = estimated_cost(usage)
        estimate = estimated_cost(usage, peak_at(datetime.now(timezone.utc)))
        with self.lock:
            if ident not in self.pending or ident in self.settled:
                raise StopCollection("invalid_or_duplicate_billing_settlement")
            reserved = self.pending[ident]
            event = {"type": "settle", "id": ident, "at": g.utc(), "usage": usage,
                     "estimated_usd": estimate, "ceiling_usd": ceiling}
            append_event(self.log, event)
            self.apply(event)
            if ceiling > reserved + 1e-12:
                raise StopCollection("provider_usage_exceeded_conservative_reservation")
        return estimate, ceiling

    def metrics(self):
        with self.lock:
            return {"estimated_cost_usd": round(self.estimated_usd, 6),
                    "settled_cost_ceiling_usd": round(self.ceiling_usd, 6),
                    "cost_ceiling_usd": round(self.ceiling_usd + sum(self.pending.values()), 6),
                    "reserved_cost_usd": round(sum(self.pending.values()), 6),
                    "pending_api_calls": len(self.pending), "budget_usd": self.budget,
                    "api_calls": self.calls, "token_usage": dict(self.usage),
                    "pricing_source": PRICING, "pricing_checked_at": "2026-09-08",
                    "cost_is_estimate_not_invoice": True}


class Client:
    def __init__(self, root, budget, stop_check):
        self.root, self.stop_check = root, stop_check
        self.ledger = Ledger(root, budget)
        self.key = os.environ.pop("DEEPSEEK_API_KEY")

    def chat(self, turns, schema, seed, *, role, candidate, checkpoint, max_tokens=2048):
        turns = [dict(turn) for turn in turns]
        turns[0]["content"] = turns[0]["content"].replace("/no_think", "")
        turns[0]["content"] += "\nReturn only a JSON object matching this JSON Schema: " + json.dumps(schema)
        payload = {"model": MODEL, "messages": turns, "stream": False,
                   "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
                   "max_tokens": max_tokens, "temperature": .75 if role == "author" else .2}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_hash = g.digest(payload)
        call = candidate.get(role + "_call")
        if call:
            if call["request_sha256"] != request_hash:
                raise StopCollection("checkpoint_request_mismatch")
            cached = self.root / "responses" / (call["id"] + ".json")
            if call["id"] in self.ledger.settled and cached.exists():
                saved = g.read_json(cached)
                result, receipt = saved["result"], saved["receipt"]
                if receipt["request_sha256"] != request_hash or receipt["response_sha256"] != g.digest(result):
                    raise StopCollection("cached_response_integrity_mismatch")
                return self.parse(result, receipt, schema, role)
            # A crash before reserve is safe to retry. A settled call without its
            # cached response is not: do not silently pay twice for it.
            if call["id"] in self.ledger.settled:
                raise StopCollection("settled_response_missing_inspect_before_resume")
        self.stop_check()
        ident = uuid.uuid4().hex
        candidate[role + "_call"] = {"id": ident, "request_sha256": request_hash}
        checkpoint(candidate)
        reserve = ((len(body) + 2048) * .44 + max_tokens * 1.32) / 1_000_000
        self.ledger.reserve(reserve, ident, self.stop_check, role=role,
                            spec_id=candidate["spec"]["id"], request_sha256=request_hash)
        started_at, started = g.utc(), time.perf_counter()
        request = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        # Each worker owns its HTTP opener; only ledger mutations are serialized.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=180) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise StopCollection("provider_http_" + str(error.code) + "_billing_unresolved") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise StopCollection("provider_transport_error_billing_unresolved") from None
        usage = result.get("usage")
        if not isinstance(usage, dict) or "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise StopCollection("provider_usage_missing_billing_unresolved")
        counts = {k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens")}
        # Save the response before settlement. An interrupted reservation remains
        # blocked for inspection, with the response available to reconcile it.
        receipt = {"at": g.utc(), "started_at": started_at, "call_id": ident, "role": role,
                   "request_sha256": request_hash, "response_sha256": g.digest(result),
                   "seconds": time.perf_counter() - started, "usage": usage,
                   "provider": "deepseek_direct", "model": result.get("model"),
                   "response_id": result.get("id"), "system_fingerprint": result.get("system_fingerprint"),
                   "variation_seed": seed, "server_seed_supported": False,
                   "spec_id": candidate["spec"]["id"]}
        if any(type(counts[k]) is not int or counts[k] < 0 for k in counts):
            raise StopCollection("invalid_provider_usage_billing_unresolved")
        receipt.update(ceiling_usd=estimated_cost(counts),
                       estimated_usd=estimated_cost(counts, peak_at(datetime.now(timezone.utc))))
        durable_json(self.root / "responses" / (ident + ".json"), {"result": result, "receipt": receipt})
        self.ledger.settle(ident, counts)
        with self.ledger.lock:
            append_event(self.root / "requests.jsonl", receipt)
        return self.parse(result, receipt, schema, role)

    @staticmethod
    def parse(result, receipt, schema, role):
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("incomplete_" + role)
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("missing_" + role + "_content")
        parsed = json.loads(content, object_pairs_hook=g.unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        schema_check(parsed, schema)
        return parsed, {**receipt, "raw_content": content}


def make_contract():
    return {"version": VERSION, "model": MODEL, "expected_revision": REVISION,
            "code_sha256": g.sha(__file__), "protocol_code_sha256": g.sha(OUT / "protocol.py"),
            "validation_code_sha256": g.sha(LOCAL / "generate.py"),
            "normalizer_code_sha256": g.sha(OUT / "validation.py"),
            "audience_code_sha256": g.sha(OUT / "audience.py"),
            "budget_origin_sha256": g.sha(OUT / "budget-origin.json"),
            "migration_sha256": g.sha(OUT / "migration.json"), "specs_sha256": g.digest(specifications()),
            "runtime_revision":"compact-blind-verifier-no-narrative-v1",
            "verifier_code_sha256":g.sha(OUT / "compact_verifier.py"),
            "target": 5000, "target_splits": {"train":4000, "validation":500, "final_test":500},
            "curriculum": audience.VERSION, "max_workers": 4,
            "output_permission": TERMS, "clause": "4.2(3)", "checked_at": "2026-09-08"}


def load_accepted(root, contract, spec_map):
    migration = g.read_json(root / "migration.json")
    source = root.parent / migration["source_corpus"]
    if not (source / "PAUSE").exists() or g.sha(source / "contract.json") != migration["parent_contract_file_sha256"]:
        raise StopCollection("parent_corpus_changed_or_not_paused")
    accepted, duplicates = {}, g.DuplicateIndex()
    for path in sorted((root / "accepted").glob("*.json")):
        saved = g.read_json(path); ep = saved["example"]
        imported_hash = migration["imported_files"].get(path.name)
        if imported_hash:
            if g.sha(path) != imported_hash or g.sha(source / "accepted" / path.name) != imported_hash:
                raise StopCollection("imported_receipt_changed")
        elif saved["contract_sha256"] != g.digest(contract):
            raise StopCollection("committed_contract_mismatch")
        if (saved["example_sha256"] != g.digest(ep) or path.stem != ep["id"] or ep["id"] in accepted
                or ep["provenance"]["spec_sha256"] != g.digest(spec_map[ep["id"]])):
            raise StopCollection("committed_example_integrity_mismatch")
        duplicates.add(duplicates.check(ep)); accepted[ep["id"]] = ep
    if not set(migration["imported_files"]).issubset(p.name for p in (root / "accepted").glob("*.json")):
        raise StopCollection("imported_example_missing")
    return accepted, duplicates


class Collector:
    def __init__(self, root, args, contract, specs, accepted, duplicates):
        self.root, self.args, self.contract, self.specs = root, args, contract, specs
        self.accepted, self.duplicates = accepted, duplicates
        self.lock, self.stop_event = threading.RLock(), threading.Event()
        self.reason, self.active = None, set()
        self.existing, self.started = len(accepted), time.monotonic()
        self.attempts, self.rejections, self.resumed = 0, Counter(), 0
        self.client = None
        self.previous = Counter()
        for path in (root / "rejected").glob("*.json"):
            saved = g.read_json(path)
            self.previous[saved["spec"]["id"]] = max(self.previous[saved["spec"]["id"]], saved["attempt"] + 1)

    def stop(self, reason):
        with self.lock:
            if self.reason is None:
                self.reason = reason
            self.stop_event.set()

    def check_stop(self):
        # No collector lock here: API reservations hold the ledger lock.
        if self.stop_event.is_set():
            raise StopCollection(self.reason or "stopped")
        if (self.root / "PAUSE").exists():
            raise StopCollection("pause_requested")
        if time.monotonic() - self.started >= self.args.max_hours * 3600:
            raise StopCollection("time_limit")

    def checkpoint(self, candidate):
        durable_json(self.root / "pending" / (candidate["spec"]["id"] + ".json"), candidate)

    def status(self, state="running", phase=None, **extra):
        with self.lock:
            elapsed = time.monotonic() - self.started
            durable_json(self.root / "status.json", {
                "state": state, "phase": phase or ("draining_inflight" if self.stop_event.is_set() else "generating_parallel"),
                "updated_at": g.utc(), "pid": os.getpid(), "runtime_pid": None,
                "target": 5000, "accepted_examples":len(self.accepted),
                "accepted_this_run":len(self.accepted)-self.existing, "attempts_this_run":self.attempts,
                "rejections_this_run":dict(self.rejections), "workers":self.args.workers,
                "active_candidates":len(self.active), "pending_checkpoints":len(list((self.root / "pending").glob("*.json"))),
                "resumed_candidates_this_run":self.resumed, "elapsed_seconds":elapsed,
                "examples_per_hour_this_run":(len(self.accepted)-self.existing)*3600/max(.001,elapsed),
                "splits":dict(Counter(e["split"] for e in self.accepted.values())),
                "mode":"collection", "limit_hours":self.args.max_hours, "limit_new_examples":self.args.max_new,
                "teacher":MODEL, "execution":"deepseek_direct_api", "curriculum":audience.VERSION,
                "audience_label":"Entrepreneurs and freelancers",
                "audience_counts":dict(Counter(e["research_context"]["audience"] for e in self.accepted.values())),
                "scenario_counts":dict(Counter(e["research_context"]["scenario"] for e in self.accepted.values())),
                "preserved_previous_examples":109, "previous_examples_in_current_count":False,
                "imported_accepted_examples":len(g.read_json(self.root / "migration.json")["imported_files"]),
                **(self.client.ledger.metrics() if self.client else {}), **extra})

    def commit(self, candidate):
        ep, ident = candidate["example"], candidate["spec"]["id"]
        with self.lock:
            if len(self.accepted) - self.existing >= self.args.max_new:
                raise StopCollection("new_example_limit")
            path = self.root / "accepted" / (ident + ".json")
            if ident in self.accepted or path.exists():
                raise StopCollection("refusing_to_overwrite_accepted_example")
            fingerprint = self.duplicates.check(ep)
            candidate.update(example_sha256=g.digest(ep), accepted_at=g.utc())
            durable_json(path, candidate)
            self.duplicates.add(fingerprint)
            self.accepted[ident] = ep
            (self.root / "pending" / (ident + ".json")).unlink(missing_ok=True)

    def process(self, spec):
        ident = spec["id"]
        path = self.root / "pending" / (ident + ".json")
        candidate = g.read_json(path) if path.exists() else None
        if candidate:
            if candidate["spec"] != spec or candidate["contract_sha256"] != g.digest(self.contract):
                raise StopCollection("pending_checkpoint_contract_mismatch")
            with self.lock:
                self.resumed += 1
                self.previous[ident] = max(self.previous[ident], candidate["attempt"] + 1)
        for _ in range(4):
            self.check_stop()
            if candidate is None:
                with self.lock:
                    attempt = self.previous[ident]
                    self.previous[ident] += 1
                candidate = {"spec":spec, "attempt":attempt, "created_at":g.utc(),
                             "contract_sha256":g.digest(self.contract)}
                self.checkpoint(candidate)
            with self.lock:
                self.attempts += 1
            try:
                attempt = candidate["attempt"]
                if "authored" not in candidate:
                    authored, ar = self.client.chat(audience.author_messages(g, spec, attempt), g.author_schema(spec),
                        spec["seed"]+attempt, role="author", candidate=candidate, checkpoint=self.checkpoint)
                    candidate.update(authored=authored, author_receipt=ar)
                    self.checkpoint(candidate)
                ep = validation.normalize(candidate["authored"], spec, candidate["author_receipt"])
                ep["provenance"].update(type="hosted_model_authored_hybrid", version=VERSION,
                    teacher=MODEL, teacher_execution="deepseek_direct_api", model_revision=REVISION,
                    model_reported=candidate["author_receipt"]["model"], intent_generator="teacher_task_descriptions_joined_v2",
                    output_permission=TERMS, output_permission_clause="4.2(3)")
                ep["reference_status"] = "synthetic_author_membership_pending_blind_regroup"
                ep["limitations"] = ["synthetic_sessions", "same_model_author_and_blind_verifier",
                    "not_independently_human_reviewed", "explicit_intent_guided_grouping",
                    "invented_profiles_and_sources_not_scraped_pages", "shared_prompt_across_splits"]
                candidate["example"] = ep
                with self.lock:
                    self.duplicates.check(ep)
                g.validate_corpus([ep])
                if "blind_review" not in candidate:
                    checked, vr = self.client.chat(verifier.messages(ep), verifier.schema(g, ep), spec["seed"]+104729,
                        role="verifier", max_tokens=1024, candidate=candidate, checkpoint=self.checkpoint)
                    candidate.update(blind_review=verifier.normalize(g, schema_check, ep, checked), compact_review=checked, review_receipt=vr)
                    self.checkpoint(candidate)
                checked, vr = candidate["blind_review"], candidate["review_receipt"]
                agreement = g.verify_grouping(ep, checked)
                ep["reference_status"] = "synthetic_author_and_blind_regroup_agree_not_human_reviewed"
                ep["provenance"]["verifier_request_sha256"] = vr["request_sha256"]
                ep["teacher_review"] = {"ambiguous_refs":[], "review_notes":checked["review_notes"], "blind_partition_agreement":agreement}
                candidate["example"] = ep
                self.checkpoint(candidate)
                self.commit(candidate)
                return
            except (ValueError, KeyError, TypeError) as error:
                reason = str(error)[:160]
                candidate["rejection"] = reason
                durable_json(self.root / "rejected" / f"{ident}-{candidate['attempt']}-{time.time_ns()}.json", candidate)
                path.unlink(missing_ok=True)
                candidate = None
                with self.lock:
                    self.rejections[reason] += 1

    def worker(self, spec):
        try:
            self.process(spec)
        except StopCollection as error:
            self.stop(str(error))
        except BaseException:
            self.stop("worker_failed_inspect_logs")
            raise

    def run(self):
        failure = None
        try:
            self.client = Client(self.root, self.args.budget_usd, self.check_stop)
            with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
                for _ in range(3):
                    # Resume partially paid candidates before starting fresh ones.
                    queue = deque(sorted((s for s in self.specs if s["id"] not in self.accepted),
                        key=lambda s: not (self.root / "pending" / (s["id"] + ".json")).exists()))
                    futures = {}
                    while queue or futures:
                        try:
                            self.check_stop()
                            if len(self.accepted)-self.existing >= self.args.max_new:
                                raise StopCollection("new_example_limit")
                        except StopCollection as error:
                            self.stop(str(error))
                        while queue and len(futures) < self.args.workers and not self.stop_event.is_set():
                            spec = queue.popleft()
                            with self.lock:
                                self.active.add(spec["id"])
                            futures[executor.submit(self.worker, spec)] = spec["id"]
                        try:
                            self.status()
                        except BaseException:
                            self.stop("status_write_failed_inspect_logs")
                            raise
                        if not futures:
                            break
                        done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                        for future in done:
                            with self.lock:
                                self.active.discard(futures.pop(future))
                            try:
                                future.result()
                            except BaseException as error:
                                failure = type(error).__name__
                                self.stop("worker_failed_inspect_logs")
                    if self.stop_event.is_set():
                        break
            if failure:
                raise RuntimeError(failure)
            state = "completed" if len(self.accepted) == 5000 else "paused"
            self.status(state, self.reason or ("finished" if state == "completed" else "attempt_limit"),
                        snapshot=export_snapshot(self.accepted))
        except StopCollection as error:
            self.status("paused", str(error), snapshot=export_snapshot(self.accepted))
        except BaseException as error:
            # The executor context has drained before this terminal status.
            self.active.clear()
            self.status("failed", "error", error=type(error).__name__)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-usd", type=float, default=8)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--workers", type=int, choices=range(1,5), default=4)
    parser.add_argument("--max-new", type=int, default=5000)
    args = parser.parse_args()
    if not 0 < args.budget_usd <= 8 or not 0 < args.max_hours <= 24 or not 0 < args.max_new <= 5000:
        parser.error("Run limits exceeded")
    for name in ("accepted", "rejected", "pending", "responses", "snapshots", "logs"):
        (OUT / name).mkdir(exist_ok=True)
    with collector_lock():
        audience.verify_budget_origin(g)
        contract = make_contract()
        if (OUT / "contract.json").exists() and g.read_json(OUT / "contract.json") != contract:
            raise StopCollection("Frozen contract changed; inspect before resuming")
        durable_json(OUT / "contract.json", contract)
        specs = specifications()
        accepted, duplicates = load_accepted(OUT, contract, {s["id"]:s for s in specs})
        # A commit can survive a crash just before its redundant checkpoint is removed.
        for ident in accepted:
            (OUT / "pending" / (ident + ".json")).unlink(missing_ok=True)
        Collector(OUT, args, contract, specs, accepted, duplicates).run()


if __name__ == "__main__":
    main()
