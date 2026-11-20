"""Bounded sequential generation, validation, revision and reproducible run records."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import TypeVar
from uuid import uuid4
from pydantic import ValidationError

from .checks import cited_source_ids, plan_duplicate_diagnostics, validate_draft, validate_plan
from . import __version__
from .interfaces import Backend, GenerationRequest
from .io import atomic_json, parse_json_object, text_hash
from .models import Draft, Model, Plan, Review, Selection, Settings, State, Task
from .prompts import messages

T = TypeVar("T", bound=Model)


class RunHalted(RuntimeError):
    def __init__(self, message: str, status: str = "failed"):
        super().__init__(message)
        self.status = status


class GenerationFormatError(RunHalted):
    """A bounded proposal failed its output contract; backend/budget still usable."""


class Budget:
    """Admission reserves a full request; accounting includes malformed generations."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.accounting: set[str] = set()
        self.unaccounted_calls = 0

    def admit(self, prompt_tokens: int, max_new_tokens: int, context_limit: int | None = None):
        context_limit = context_limit or self.settings.max_context_tokens
        if prompt_tokens + max_new_tokens > context_limit:
            raise RunHalted(
                f"Context requires up to {prompt_tokens + max_new_tokens} tokens, limit is "
                f"{context_limit}. Reduce the source packet or output budget; "
                "the runner will not silently truncate evidence."
            )
        if self.calls >= self.settings.max_calls:
            raise RunHalted("Model call budget exhausted", "budget_exhausted")
        if self.output_tokens + max_new_tokens > self.settings.max_output_tokens:
            raise RunHalted("Insufficient remaining generated-token budget for this call", "budget_exhausted")
        if self.input_tokens + self.output_tokens + prompt_tokens + max_new_tokens > self.settings.max_run_tokens:
            raise RunHalted("Insufficient remaining total-token budget for this call", "budget_exhausted")
        self.calls += 1

    def record(self, result):
        if type(result.input_tokens) is not int or type(result.output_tokens) is not int:
            raise RunHalted("Backend returned invalid token counts")
        if result.input_tokens < 0 or result.output_tokens < 0:
            raise RunHalted("Backend returned negative token counts")
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.accounting.add(result.accounting)

    def to_dict(self):
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "accounting": sorted(self.accounting),
                "unaccounted_calls": self.unaccounted_calls,
                "complete": self.unaccounted_calls == 0}


class Pipeline:
    def __init__(self, backend: Backend, settings: Settings, output_dir: str | Path, *, run_id: str | None = None, source_manifest: dict | None = None):
        self.backend = backend
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.run_id = run_id or uuid4().hex
        self.source_manifest = source_manifest
        self.budget = Budget(settings)
        self._draft: Draft | None = None
        self._task: Task | None = None
        self._event_sequence = 0

    def _event(self, kind: str, **payload):
        self._event_sequence += 1
        event = {"sequence": self._event_sequence, "time": datetime.now(timezone.utc).isoformat(),
                 "kind": kind, **payload}
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def _call(self, role: str, expected: type[T], *, max_new_tokens: int, **kwargs) -> T:
        assert self._task is not None
        prompt_messages = messages(role, self._task, **kwargs)
        last_error = ""
        for retry in range(self.settings.schema_retries + 1):
            current = list(prompt_messages)
            if last_error:
                current.append({"role": "user", "content": "The previous attempt failed validation: " + last_error[:1200] + ". Return a fresh complete JSON object that obeys the schema. Keep it compact; do not omit substantive evidence."})
            thinking = (self.settings.thinking_plan if role == "plan" else
                        self.settings.thinking_review if role.startswith("review") else False)
            temperature = (self.settings.plan_temperature if role == "plan" else
                           self.settings.review_temperature if role.startswith("review") or role in {"select", "semantic_critic"} else
                           self.settings.draft_temperature)
            request = GenerationRequest(
                role=role, messages=current, max_new_tokens=max_new_tokens,
                temperature=temperature, seed=self.settings.seed + self.budget.calls,
                top_p=self.settings.thinking_top_p if thinking else self.settings.nonthinking_top_p,
                top_k=self.settings.top_k, min_p=self.settings.min_p,
                thinking=thinking, metadata={"task": self._task.model_dump(), **kwargs,
                    "use_adapter": role in {"plan", "draft", "repair"}},
            )
            prompt_tokens = self.backend.count_input_tokens(request)
            critic_context = (self.settings.critic_max_context_tokens if self.settings.critic_model_id and role in {"semantic_critic", "review_evidence"} else None)
            self.budget.admit(prompt_tokens, max_new_tokens, context_limit=critic_context)
            call_id = self.budget.calls
            self._event("request", call_id=call_id, request=asdict(request), estimated_input_tokens=prompt_tokens)
            started = time.perf_counter()
            try:
                result = self.backend.generate(request)
            except BaseException as exc:
                self.budget.unaccounted_calls += 1
                self._event("backend_error", call_id=call_id, error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter()-started)
                if not isinstance(exc, Exception):
                    raise
                raise RunHalted(f"Backend failed during {role}: {exc}") from exc
            self._event("response", call_id=call_id, response=asdict(result))
            try:
                self.budget.record(result)
            except RunHalted:
                self.budget.unaccounted_calls += 1
                raise
            if result.output_tokens > max_new_tokens:
                raise RunHalted("Backend exceeded the reserved output-token limit")
            if result.input_tokens != prompt_tokens and result.accounting == "tokenizer":
                raise RunHalted("Backend tokenizer accounting changed between admission and generation")
            if self.budget.output_tokens > self.settings.max_output_tokens or self.budget.input_tokens + self.budget.output_tokens > self.settings.max_run_tokens:
                raise RunHalted("Backend accounting exceeded the run budget", "budget_exhausted")
            try:
                if result.finish_reason != "stop":
                    raise ValueError(f"Generation did not finish normally ({result.finish_reason}); incomplete output is not accepted")
                if result.metadata.get("complete") is False:
                    raise ValueError("Backend marked this generation incomplete")
                value = expected.model_validate(parse_json_object(result.text))
                if role == "plan" and value.id != kwargs.get("plan_id"):
                    raise ValueError(f"Plan ID must be {kwargs.get('plan_id')}")
                return value
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                self._event("schema_error", call_id=call_id, error=last_error)
        raise GenerationFormatError(f"{role} did not produce a valid complete {expected.__name__}: {last_error[:500]}")

    def _plans(self, task: Task):
        if self.settings.strategy == "direct":
            return None, []
        count = 1 if self.settings.strategy == "single_plan" else self.settings.candidates
        candidates = []
        previous = []
        atomic_json(self.output_dir / "accepted_plans.json", candidates)
        for index in range(count):
            plan_id = f"P{index+1}"
            try:
                plan = self._call("plan", Plan, max_new_tokens=self.settings.plan_tokens,
                                  previous=previous, plan_id=plan_id)
            except GenerationFormatError as exc:
                self._event("candidate_format_failure", plan_id=plan_id, stage="plan", error=str(exc))
                continue
            previous.append({"approach": plan.approach[:350], "intent": plan.intent[:350],
                             "opening_moves": [move.description[:350] for move in plan.moves[:3]]})
            plan_dict = plan.model_dump()
            findings = validate_plan(task.model_dump(), plan_dict)
            self._event("plan_checks", plan_id=plan.id, findings=findings)
            if any(i["severity"] == "material" for i in findings):
                self._event("plan_rejected", plan=plan_dict, reason="deterministic_checks")
                continue
            try:
                review = self._call("review_plan", Review, max_new_tokens=self.settings.review_tokens,
                                    plan=plan_dict, defects=findings)
            except GenerationFormatError as exc:
                self._event("candidate_format_failure", plan_id=plan_id, stage="review_plan", error=str(exc))
                continue
            self._event("plan_review", plan_id=plan.id, review=review.model_dump())
            if review.decision == "accept":
                candidates.append({"plan": plan_dict, "review": review.model_dump()})
                atomic_json(self.output_dir / "accepted_plans.json", candidates)
        atomic_json(self.output_dir / "accepted_plans.json", candidates)
        self._event("plan_diversity_diagnostics", findings=plan_duplicate_diagnostics([row["plan"] for row in candidates]))
        if not candidates:
            raise RunHalted("No plan passed checks. Inspect plan findings; do not invent evidence to force an answer.")
        if len(candidates) == 1:
            selection = {"plan_id": candidates[0]["plan"]["id"], "reason": "Only one candidate passed checks"}
        elif self.settings.selection_policy == "sample":
            selection_seed = self.settings.seed + 100003
            chosen_index = random.Random(selection_seed).randrange(len(candidates))
            selection = {"plan_id": candidates[chosen_index]["plan"]["id"], "reason": "Seeded uniform sample among accepted candidates; diversity ablation"}
            self._event("selection_policy", policy="sample", seed=selection_seed)
        else:
            shuffled = list(candidates)
            selection_seed = self.settings.seed + 100003
            random.Random(selection_seed).shuffle(shuffled)
            aliases = {f"C{index+1}": row["plan"]["id"] for index, row in enumerate(shuffled)}
            shown = [{"plan": {**row["plan"], "id": alias}} for alias, row in zip(aliases, shuffled)]
            self._event("selection_context", policy="judge", seed=selection_seed, aliases=aliases)
            result = self._call("select", Selection, max_new_tokens=self.settings.selection_tokens,
                                candidates=shown).model_dump()
            if result["plan_id"] not in aliases:
                raise RunHalted("Selector named a plan outside its blinded candidate set")
            selection = {**result, "plan_id": aliases[result["plan_id"]]}
        by_id = {row["plan"]["id"]: row["plan"] for row in candidates}
        if selection["plan_id"] not in by_id:
            raise RunHalted("Selector named a plan that did not pass checks")
        self._event("selection", selection=selection)
        chosen = by_id[selection["plan_id"]]
        atomic_json(self.output_dir / "selected_plan.json", chosen)
        return chosen, candidates

    def _draft_findings(self, task: Task, draft: Draft, plan):
        data = draft.model_dump()
        issues = validate_draft(task.model_dump(), data, plan=plan)
        if task.mode == "history" and (draft.state_updates.world or draft.state_updates.reader):
            issues.append({"code": "history_state_mutation", "message": "Historical sources and factual context cannot be rewritten by a draft.", "severity": "material"})
        return issues

    def _write_references(self, task: Task, text: str) -> None:
        """Render only supplied bibliographic metadata, never model inventions."""
        source_map = {source.id: source for source in task.sources}
        cited = cited_source_ids(text)
        records = []
        lines = ["References from supplied metadata", "",
                 "Entries preserve supplied titles, dates, locators and URLs; external authenticity is not verified.", ""]
        for identifier in cited:
            if identifier not in source_map:
                continue
            source = source_map[identifier]
            data = source.model_dump(exclude={"text"})
            data["passage_sha256"] = text_hash(source.text)
            records.append(data)
            fields = [source.author, source.title, source.date, source.locator, source.url]
            lines.append(f"[{identifier}] " + "; ".join(value for value in fields if value and value != "unspecified"))
        unknown = [identifier for identifier in cited if identifier not in source_map]
        if unknown:
            lines.extend(["", "Unresolved source IDs: " + ", ".join(unknown)])
        atomic_json(self.output_dir / "references.json", {
            "output_sha256": text_hash(text), "references": records,
            "unknown_source_ids": unknown,
            "supplied_but_not_cited": [identifier for identifier in source_map if identifier not in cited],
            "source_scope": task.source_scope,
            "external_authenticity_verified": False,
        })
        (self.output_dir / "references.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def assess(self, task: Task, draft_text: str) -> dict:
        """Critique supplied text, without representing it as a generated output.

        A valid report means the critic obeyed its evidence/report contract. It
        does not mean the draft was accepted, or its claims independently verified.
        """
        from .critic import CriticReport, validate_critic_report
        if not isinstance(draft_text, str) or not draft_text.strip():
            raise ValueError("A critic assessment needs a nonempty draft")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._task = task
        started = time.perf_counter()
        atomic_json(self.output_dir / "task.json", task.model_dump())
        atomic_json(self.output_dir / "config.json", self.settings.model_dump())
        atomic_json(self.output_dir / "input_draft.json", {"text": draft_text, "origin": "user_supplied"})
        if self.source_manifest is not None:
            atomic_json(self.output_dir / "source_selection.json", self.source_manifest)
        result = {"pipeline_version": __version__, "assessment_id": self.run_id,
                  "task_id": task.task_id, "mode": task.mode,
                  "status": "failed", "input_sha256": text_hash(draft_text),
                  "is_fixture": self.settings.backend == "demo", "error": None,
                  "report": None, "validation": None,
                  "meaning": "Report validation checks traceability and contract consistency, not factual truth or literary quality."}
        try:
            report = self._call("semantic_critic", CriticReport, max_new_tokens=self.settings.critic_tokens,
                                draft={"text": draft_text}).model_dump()
            validation = validate_critic_report(task.model_dump(), {"text": draft_text}, report)
            result.update(report=report, validation=validation,
                          status="invalid_report" if any(row["severity"] == "material" for row in validation) else "valid_report")
            atomic_json(self.output_dir / "critic.json", report)
            self._event("critic_assessment", input_sha256=result["input_sha256"], report=report, validation=validation)
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Critic assessment interrupted")
            raise
        except RunHalted as exc:
            result.update(status=exc.status, error=str(exc))
        except Exception as exc:
            result.update(status="failed", error=f"Unexpected {type(exc).__name__}: {exc}")
        finally:
            result["tokens"] = self.budget.to_dict()
            result["elapsed_seconds"] = time.perf_counter() - started
            result["backend"] = self.backend.describe()
            atomic_json(self.output_dir / "assessment.json", result)
        return result

    def run(self, task: Task) -> dict:
        """Write one exclusive run directory. Never overwrite an earlier experiment."""
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._task = task
        started = time.perf_counter()
        atomic_json(self.output_dir / "task.json", task.model_dump())
        atomic_json(self.output_dir / "config.json", self.settings.model_dump())
        if self.source_manifest is not None:
            atomic_json(self.output_dir / "source_selection.json", self.source_manifest)
        summary = {"pipeline_version": __version__, "run_id": self.run_id, "task_id": task.task_id, "mode": task.mode, "strategy": self.settings.strategy,
                   "backend": self.backend.describe(), "status": "failed", "is_fixture": self.settings.backend == "demo",
                   "error": None, "final_issues": [], "review_label_meaning": "Fallible model and mechanical checks, not human approval or certified truth"}
        final_state = task.initial_state.model_dump()
        try:
            plan, candidates = self._plans(task)
            summary["accepted_plan_count"] = len(candidates)
            draft = self._call("draft", Draft, max_new_tokens=self.settings.draft_tokens, plan=plan)
            self._draft = draft
            for revision in range(self.settings.max_repairs + 1):
                atomic_json(self.output_dir / f"draft_{revision:02d}.json", draft.model_dump())
                summary["draft_revision"] = revision
                summary["repairs"] = revision
                summary["final_review"] = None
                summary["reviewed_output_sha256"] = None
                summary["final_evidence_review"] = None
                summary["evidence_reviewed_output_sha256"] = None
                summary["final_critic"] = None
                summary["critic_reviewed_output_sha256"] = None
                summary["critic_validation"] = None
                mechanical = self._draft_findings(task, draft, plan)
                summary["final_issues"] = mechanical
                audit = None
                critic = None
                critic_issues = []
                if self.settings.semantic_critic:
                    from .critic import CriticReport, validate_critic_report
                    critic = self._call("semantic_critic", CriticReport, max_new_tokens=self.settings.critic_tokens,
                                        draft={"text": draft.text})
                    critic_data = critic.model_dump()
                    findings = validate_critic_report(task.model_dump(), draft.model_dump(), critic_data)
                    atomic_json(self.output_dir / f"critic_{revision:02d}.json", critic_data)
                    summary["final_critic"] = critic_data
                    summary["critic_reviewed_output_sha256"] = text_hash(draft.text)
                    summary["critic_validation"] = findings
                    self._event("semantic_critic", revision=revision, report=critic_data, validation=findings,
                                output_sha256=text_hash(draft.text))
                    if any(row["severity"] == "material" for row in findings):
                        summary["status"] = "needs_review"
                        summary["final_issues"] = mechanical + findings
                        summary["error"] = "The critic's own report failed checks; its claims are not used to rewrite the draft."
                        break
                    critic_issues = [row.model_dump() for row in critic.issues]
                    summary["final_issues"] = mechanical + critic_issues
                    if critic.decision == "abstain":
                        summary["status"] = "needs_sources" if task.mode == "history" else "needs_review"
                        summary["error"] = "The critic requested evidence or context the supplied packet cannot provide."
                        break
                elif task.mode == "history" and self.settings.history_evidence_audit:
                    audit = self._call("review_evidence", Review, max_new_tokens=self.settings.review_tokens,
                                       draft={"text": draft.text})
                    summary["final_evidence_review"] = audit.model_dump()
                    summary["evidence_reviewed_output_sha256"] = text_hash(draft.text)
                    self._event("evidence_review", revision=revision, review=audit.model_dump(), output_sha256=text_hash(draft.text))
                audit_issues = [issue.model_dump() for issue in audit.issues] if audit else []
                summary["final_issues"] = mechanical + audit_issues + critic_issues
                review = self._call("review_draft", Review, max_new_tokens=self.settings.review_tokens,
                                    plan=plan, draft=draft.model_dump(), defects=mechanical + audit_issues + critic_issues)
                issues = mechanical + audit_issues + critic_issues + [i.model_dump() for i in review.issues]
                summary["final_issues"] = issues
                summary["final_review"] = review.model_dump()
                summary["reviewed_output_sha256"] = text_hash(draft.text)
                self._event("draft_review", revision=revision, review=review.model_dump(), deterministic_findings=mechanical)
                if review.decision == "accept" and (audit is None or audit.decision == "accept") and (critic is None or critic.decision == "accept") and not any(i["severity"] == "material" for i in issues):
                    summary["status"] = "accepted_by_checks"
                    if task.mode == "english":
                        for channel in ("world", "reader"):
                            known = {row["id"] for row in final_state[channel]}
                            final_state[channel].extend(row for row in draft.state_updates.model_dump()[channel] if row["id"] not in known)
                    break
                if revision == self.settings.max_repairs:
                    summary["status"] = "needs_review"
                    break
                repair_notes = list(issues)
                if critic is not None:
                    repair_notes.append({"code": "critic_editing_guidance", "severity": "minor",
                                         "message": "Use these concrete priorities while preserving effective choices unless they cause a material defect.",
                                         "revision_priorities": critic.model_dump().get("revision_priorities", []),
                                         "preserve": critic.model_dump().get("preserve", [])})
                draft = self._call("repair", Draft, max_new_tokens=self.settings.draft_tokens,
                                   plan=plan, draft=draft.model_dump(), defects=repair_notes or [{"message": "Reviewer requested revision without an explicit issue; re-evaluate the draft carefully."}])
                self._draft = draft
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            summary["error"] = "Interrupted before the run completed"
            self._event("run_halted", status="interrupted", error=summary["error"])
            raise
        except RunHalted as exc:
            summary["status"] = exc.status
            summary["error"] = str(exc)
            self._event("run_halted", status=exc.status, error=str(exc))
        except Exception as exc:
            summary["status"] = "failed"
            summary["error"] = f"Unexpected {type(exc).__name__}: {exc}"
            self._event("unexpected_error", error=summary["error"])
        finally:
            if self._draft is not None:
                atomic_json(self.output_dir / "final.json", self._draft.model_dump())
                (self.output_dir / "final.txt").write_text(self._draft.text, encoding="utf-8")
                summary["output_sha256"] = text_hash(self._draft.text)
                summary["word_count"] = len(self._draft.text.split())
                if task.mode == "history":
                    self._write_references(task, self._draft.text)
            else:
                summary["output_sha256"] = None
                summary["word_count"] = 0
            atomic_json(self.output_dir / "state.json", State.model_validate(final_state).model_dump())
            summary["tokens"] = self.budget.to_dict()
            summary["elapsed_seconds"] = time.perf_counter() - started
            summary["backend"] = self.backend.describe()
            atomic_json(self.output_dir / "summary.json", summary)
        return summary

