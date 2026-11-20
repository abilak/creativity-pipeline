"""Small, designated development checks for the prose critic.

This runner assesses supplied drafts only. It never trains a model or asks a
writer to produce one. Synthetic cases are development diagnostics, not a
held-out benchmark or evidence of general historical or literary competence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping
from uuid import uuid4

from pydantic import Field, StringConstraints, field_validator, model_validator

from .critic import CriticReport, validate_critic_report
from .demo import DemoBackend
from .evaluation import condition_provenance
from .interfaces import Backend
from .io import atomic_json, parse_json_object, text_hash
from .models import Model, Settings, Task
from .pipeline import Pipeline

Decision = Literal["accept", "revise", "reject", "abstain"]
_STATES = ("pending", "running", "complete", "failed", "interrupted")
_LIMITATION = (
    "Synthetic development checks only. A pass means a validated critic report "
    "matched the designated expectations on this case. It does not establish "
    "general quality, fairness, factual truth, or performance on held-out work. "
    "The set is small and intentionally constructed; no statistical confidence "
    "interval or population accuracy estimate is inferred."
)


class CriticCase(Model):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    description: str = Field(min_length=1)
    task: Task
    draft_text: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(min_length=1)
    expected_decisions: list[Decision] = Field(min_length=1)
    expected_issue_codes_any: list[str] = Field(default_factory=list)

    @field_validator("draft_text")
    @classmethod
    def nonblank_draft(cls, value):
        if not value.strip():
            raise ValueError("draft_text must contain non-whitespace prose")
        return value

    @model_validator(mode="after")
    def unique_expectations(self):
        if len(set(self.expected_decisions)) != len(self.expected_decisions):
            raise ValueError("expected_decisions must not contain duplicates")
        if any(not code.strip() or code != code.strip() for code in self.expected_issue_codes_any):
            raise ValueError("Expected issue codes must be nonempty, without surrounding whitespace")
        if len(set(self.expected_issue_codes_any)) != len(self.expected_issue_codes_any):
            raise ValueError("expected_issue_codes_any must not contain duplicates")
        return self


def _read_cases(path: str | Path) -> tuple[list[CriticCase], str]:
    raw = Path(path).read_bytes()
    rows, identifiers = [], set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = CriticCase.model_validate(parse_json_object(line))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Critic case line {number}: {exc}") from exc
        if case.case_id in identifiers:
            raise ValueError(f"Duplicate critic case_id {case.case_id!r} on line {number}")
        identifiers.add(case.case_id)
        rows.append(case)
    if not rows:
        raise ValueError("A critic evaluation needs at least one designated case")
    return rows, hashlib.sha256(raw).hexdigest()


def load_critic_cases(path: str | Path) -> list[CriticCase]:
    """Validate all JSONL cases before any output directory or model call."""
    return _read_cases(path)[0]


def _has_fixture_metadata(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(value.get(key) for key in ("fixture", "is_fixture")):
            return True
        if value.get("backend") in ("demo", "fixture"):
            return True
        return any(_has_fixture_metadata(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_fixture_metadata(child) for child in value)
    return False


def _time() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure(kind: str, message: str) -> dict[str, str]:
    return {"type": kind, "message": message}


def _record_assessment(row: dict, case: CriticCase, assessment: Any, settings: Settings) -> None:
    """Retain the assessment and score only a traceable, valid report."""
    row["assessment"] = assessment
    if not isinstance(assessment, Mapping):
        raise ValueError("Pipeline.assess did not return an assessment object")
    row["assessment_status"] = assessment.get("status")
    if _has_fixture_metadata(assessment):
        raise ValueError("Fixture assessments cannot be used for critic evaluation")
    description = assessment.get("backend")
    if isinstance(description, Mapping):
        row["condition"] = condition_provenance(settings.model_dump(), description)
    if assessment.get("input_sha256") != row["input_sha256"]:
        raise ValueError("The assessment input hash does not match the designated draft")
    if assessment.get("task_id") != case.task.task_id:
        raise ValueError("The assessment task_id does not match the designated task")
    if assessment.get("status") != "valid_report":
        row["status"] = "interrupted" if assessment.get("status") == "interrupted" else "failed"
        row["error"] = _failure(
            "AssessmentFailure", str(assessment.get("error") or f"Assessment status: {assessment.get('status')}")
        )
        return
    report = CriticReport.model_validate(assessment.get("report"))
    findings = validate_critic_report(case.task.model_dump(), {"text": case.draft_text}, report.model_dump())
    row["scoring_validation"] = findings
    if any(item.get("severity") == "material" for item in findings):
        row["status"] = "failed"
        row["error"] = _failure("InvalidCriticReport", "Report failed evidence/contract validation during scoring")
        return
    codes = sorted({item.code for item in report.issues})
    decision_match = report.decision in case.expected_decisions
    code_match = not case.expected_issue_codes_any or bool(set(codes) & set(case.expected_issue_codes_any))
    decision_error = None
    if not decision_match:
        if case.expected_decisions == ["accept"] and report.decision in {"revise", "reject"}:
            decision_error = "false_positive"
        elif set(case.expected_decisions) <= {"revise", "reject"} and report.decision == "accept":
            decision_error = "false_negative"
        elif report.decision == "abstain":
            decision_error = "unexpected_abstention"
        else:
            decision_error = "other_decision_mismatch"
    row.update(status="complete", error=None, outcome={
        "passed": bool(decision_match and code_match),
        "observed_decision": report.decision, "observed_issue_codes": codes,
        "decision_match": decision_match, "issue_code_match": code_match,
        "decision_error": decision_error,
    })


def _summary(schedule: dict) -> dict:
    rows = schedule["cases"]
    state_counts = Counter(row["status"] for row in rows)
    outcomes = [row["outcome"] for row in rows if row["status"] == "complete" and row["outcome"]]
    passed = sum(item["passed"] for item in outcomes)
    errors = Counter(item["decision_error"] for item in outcomes)
    counts = {
        "n_designated": len(rows), "n_attempted": len(rows) - state_counts["pending"],
        "n_valid_reports": len(outcomes), "n_missing_valid_reports": len(rows) - len(outcomes),
        "n_passed": passed, "n_expectation_failures": len(outcomes) - passed,
        "n_false_positives": errors["false_positive"], "n_false_negatives": errors["false_negative"],
        "n_unexpected_abstentions": errors["unexpected_abstention"],
        "n_decision_mismatches": sum(not item["decision_match"] for item in outcomes),
        "n_issue_code_mismatches": sum(not item["issue_code_match"] for item in outcomes),
        **{f"n_{state}": state_counts[state] for state in _STATES},
    }
    known_tokens = {key: 0 for key in ("calls", "input_tokens", "output_tokens", "total_tokens", "unaccounted_calls")}
    incomplete = 0
    for row in rows:
        assessment = row.get("assessment")
        tokens = assessment.get("tokens") if isinstance(assessment, Mapping) else None
        if not isinstance(tokens, Mapping):
            incomplete += row["status"] != "pending"
            continue
        incomplete += tokens.get("complete") is not True
        for key in known_tokens:
            value = tokens.get(key)
            if type(value) is int and value >= 0:
                known_tokens[key] += value
    return {
        "schema_version": 1, "evaluation_id": schedule["evaluation_id"],
        "status": schedule["status"], "development_only": True,
        "created_at": schedule["created_at"], "updated_at": schedule["updated_at"],
        "cases_sha256": schedule["cases_sha256"], "limitation": _LIMITATION,
        **{key: counts[key] for key in ("n_designated", "n_passed", "n_failed", "n_pending", "n_interrupted")},
        "counts": counts,
        "rates": {"pass_fraction_among_designated": passed / len(rows),
                  "pass_fraction_among_valid_reports": passed / len(outcomes) if outcomes else None},
        "error_definitions": {
            "false_positive": "Expected only accept; valid critic report instead requested revise/reject.",
            "false_negative": "Expected only revise/reject; valid critic report instead accepted.",
            "missing_valid_report": "No valid scored report, including pending, interrupted, failed, and invalid assessments.",
        },
        "token_accounting": {"known_totals": known_tokens, "incomplete_assessments": incomplete,
                             "note": "Known recorded totals only; unsuccessful backend calls may be unaccounted. Budgets apply per case."},
        "cases": [{key: row.get(key) for key in (
            "case_id", "description", "status", "assessment_status", "assessment_dir", "outcome", "error", "condition",
        )} for row in rows],
    }


def evaluate_critic_cases(
    cases_path: str | Path, backend: Backend, settings: Settings, output: str | Path,
) -> dict:
    """Assess a predeclared JSONL development set into a new exclusive directory.

    ``schedule.json`` contains every case and its exact inputs before the first
    assessment. Each case uses a fresh Pipeline budget and only ``assess``.
    Invalid reports and ordinary failures remain recorded while later cases
    continue. Interruptions persist the active case and all pending cases, then
    re-raise; a subsequent invocation must use a new output directory.
    """
    cases, cases_hash = _read_cases(cases_path)
    description = backend.describe()
    if not isinstance(description, Mapping):
        raise ValueError("The critic backend must provide a description object")
    if settings.backend == "demo" or isinstance(backend, DemoBackend) or _has_fixture_metadata(description):
        raise ValueError("Demo/fixture backends cannot be used for critic evaluation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    now, identifier = _time(), uuid4().hex
    schedule = {
        "schema_version": 1, "evaluation_id": identifier, "status": "running", "development_only": True,
        "created_at": now, "updated_at": now, "cases_path": str(Path(cases_path).resolve()),
        "cases_sha256": cases_hash, "configuration": settings.model_dump(),
        "backend_before": dict(description), "limitation": _LIMITATION,
        "cases": [{
            **case.model_dump(), "input_sha256": text_hash(case.draft_text),
            "task_sha256": text_hash(json.dumps(case.task.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))),
            "assessment_dir": f"assessments/{index:03d}-{case.case_id}",
            "status": "pending", "assessment_status": None, "assessment": None,
            "outcome": None, "error": None, "started_at": None, "completed_at": None,
        } for index, case in enumerate(cases, 1)],
    }

    def persist() -> dict:
        schedule["updated_at"] = _time()
        atomic_json(output / "schedule.json", schedule)
        report = _summary(schedule)
        atomic_json(output / "report.json", report)
        return report

    persist()  # All designated inputs and pending rows are durable before a call.
    for case, row in zip(cases, schedule["cases"]):
        row.update(status="running", started_at=_time())
        persist()
        directory = output / row["assessment_dir"]
        try:
            assessment = Pipeline(backend, settings, directory, run_id=f"{identifier}-{case.case_id}").assess(case.task, case.draft_text)
            _record_assessment(row, case, assessment, settings)
        except (KeyboardInterrupt, SystemExit) as exc:
            # Pipeline.assess normally saves its interrupted assessment in its
            # finally block. Keep its partial accounting if it is available.
            saved = directory / "assessment.json"
            if saved.is_file():
                try:
                    row["assessment"] = json.loads(saved.read_text(encoding="utf-8"))
                    row["assessment_status"] = row["assessment"].get("status")
                except (ValueError, OSError, AttributeError):
                    pass
            row.update(status="interrupted", error=_failure(type(exc).__name__, str(exc) or "Assessment interrupted"), completed_at=_time())
            schedule["status"] = "interrupted"
            persist()
            raise
        except Exception as exc:
            row.update(status="failed", error=_failure(type(exc).__name__, str(exc)))
        row["completed_at"] = _time()
        persist()
    schedule["status"] = "complete"  # The schedule finished, even if some cases failed.
    try:
        schedule["backend_after"] = backend.describe()
    except Exception as exc:
        schedule["backend_description_error"] = _failure(type(exc).__name__, str(exc))
    return persist()
