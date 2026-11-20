"""Local commands. Network downloads are explicit; detector queries are manual."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from uuid import uuid4

import yaml

from .io import atomic_json, load_json, parse_json_object, text_hash
from . import __version__
from .models import Draft, Plan, Settings, State, Task


def _print(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _settings(args) -> Settings:
    config_path = getattr(args, "config", None)
    raw = {}
    if config_path:
        path = Path(config_path).expanduser().resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping")
        if raw.get("adapter_path"):
            adapter = Path(raw["adapter_path"]).expanduser()
            raw["adapter_path"] = str((path.parent / adapter).resolve())
    for key in ("strategy", "seed", "critic_model_id", "critic_revision"):
        if getattr(args, key, None) is not None:
            raw[key] = getattr(args, key)
    if getattr(args, "adapter", None):
        raw["adapter_path"] = str(Path(args.adapter).expanduser().resolve())
    if getattr(args, "allow_download", False):
        raw["local_files_only"] = False
    return Settings.model_validate(raw)


def _backend(settings):
    if settings.backend == "demo":
        from .demo import DemoBackend
        return DemoBackend()
    from .backends import TransformersBackend
    from .critic_backends import SequentialCriticBackend
    writer = TransformersBackend(
        model_id=settings.model_id, revision=settings.revision, device=settings.device,
        load_in_4bit=settings.load_in_4bit, adapter_path=settings.adapter_path,
        max_context_tokens=settings.max_context_tokens,
        local_files_only=settings.local_files_only,
    )
    critic = None
    if settings.critic_model_id:
        critic = TransformersBackend(
            model_id=settings.critic_model_id, revision=settings.critic_revision,
            device=settings.device, load_in_4bit=settings.critic_load_in_4bit,
            adapter_path=None,
            max_context_tokens=settings.critic_max_context_tokens or settings.max_context_tokens,
            local_files_only=settings.local_files_only,
        )
    return SequentialCriticBackend(writer, critic)


def _task(path, state_in=None, *, source_packet=None):
    raw = load_json(Path(path).expanduser())
    if not isinstance(raw, dict):
        raise ValueError("A task must be a JSON object")
    if source_packet is not None:
        if raw.get("mode") != "history":
            raise ValueError("--source-library is for history tasks")
        from .models import Source
        merged = {}
        for row in [*raw.get("sources", []), *source_packet["sources"]]:
            source = Source.model_validate(row).model_dump()
            if source["id"] in merged and merged[source["id"]] != source:
                raise ValueError(f"Conflicting inline and retrieved source ID {source['id']!r}")
            merged[source["id"]] = source
        raw["sources"] = list(merged.values())
        manifest = source_packet["manifest"]
        scope = (
            f"Retrieved {manifest['n_selected_passages']} passages from {manifest['n_selected_documents']} "
            f"of {manifest['n_library_documents']} library documents. "
            f"{manifest['n_omitted_passages']} library passages were not selected by this retrieval. "
            "Existing inline sources were retained and may also contain material omitted by retrieval. "
            "Lexical retrieval may miss relevant or contrary evidence. Only the combined supplied packet is available "
            "for assessment; content outside it was not assessed. "
            "Document counts do not establish independent corroboration. "
            f"Document coverage shortfall: {manifest['document_coverage_shortfall']}; "
            f"uncovered query indices: {manifest['uncovered_query_indices']}."
        )
        raw["source_scope"] = "\n".join(value for value in (raw.get("source_scope", ""), scope) if value)
    raw = Task.model_validate(raw).model_dump()
    if state_in:
        if raw.get("mode") != "english":
            raise ValueError("--state-in is for English fiction only; history uses immutable sources")
        prior = State.model_validate(load_json(Path(state_in).expanduser())).model_dump()
        combined = {}
        for channel in ("world", "reader"):
            entries = {}
            for rows in (prior[channel], raw["initial_state"][channel]):
                ids = [row["id"] for row in rows]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"Imported {channel} state contains duplicate IDs")
                for row in rows:
                    if row["id"] in entries and entries[row["id"]]["text"] != row["text"]:
                        raise ValueError(f"Conflicting {channel} state for ID {row['id']!r}")
                    entries.setdefault(row["id"], row)
            combined[channel] = list(entries.values())
        raw["initial_state"] = combined
    return Task.model_validate(raw)


def _prepared_task(args):
    library = getattr(args, "source_library", None)
    packet = None
    if library:
        from .sources import retrieve_sources
        raw = load_json(Path(args.task).expanduser())
        if not isinstance(raw, dict) or raw.get("mode") != "history":
            raise ValueError("--source-library requires a history task object")
        packet = retrieve_sources(
            Path(library).expanduser(), args.source_query or raw.get("prompt", ""),
            max_chars=args.source_max_chars, max_passages=args.source_max_passages,
            per_document_limit=args.source_per_document, min_documents=args.source_min_documents,
            additional_queries=args.counter_query,
        )
    elif getattr(args, "source_query", None) or getattr(args, "counter_query", None):
        raise ValueError("Source queries require --source-library")
    task = _task(args.task, getattr(args, "state_in", None), source_packet=packet)
    return task, packet["manifest"] if packet else None


def _generate(args):
    from .pipeline import Pipeline
    settings = _settings(args)
    task, manifest = _prepared_task(args)
    summary = Pipeline(_backend(settings), settings, Path(args.output).expanduser(), source_manifest=manifest).run(task)
    _print(summary)
    return 0 if summary["status"] == "accepted_by_checks" else 2


def _build_sources(args):
    from .sources import build_source_library
    report = build_source_library(Path(args.path).expanduser(), Path(args.output).expanduser(),
                                  chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars)
    _print(report)
    return 0


def _retrieve_sources(args):
    from .sources import retrieve_sources
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a source packet: {output}")
    packet = retrieve_sources(Path(args.library).expanduser(), args.query,
                              max_chars=args.max_chars, max_passages=args.max_passages,
                              per_document_limit=args.per_document, min_documents=args.min_documents,
                              additional_queries=args.counter_query)
    atomic_json(output, packet)
    _print({"output": str(output.resolve()), **packet["manifest"]})
    return 0 if packet["sources"] else 2


def _verify_source_quote(args):
    from .sources import verify_source_quote
    result = verify_source_quote(Path(args.library).expanduser(), args.source_id, args.quote)
    _print(result)
    return 0 if result.get("valid") else 2


def _critique(args):
    from .pipeline import Pipeline
    settings = _settings(args)
    task, manifest = _prepared_task(args)
    text = Path(args.draft).expanduser().read_bytes().decode("utf-8")
    report = Pipeline(_backend(settings), settings, Path(args.output).expanduser(),
                      source_manifest=manifest).assess(task, text)
    _print(report)
    return 0 if report["status"] == "valid_report" else 2


def _evaluate_critic(args):
    from .critic_evaluation import evaluate_critic_cases
    settings = _settings(args)
    report = evaluate_critic_cases(Path(args.cases).expanduser(), _backend(settings), settings,
                                   Path(args.output).expanduser())
    _print(report)
    return 0 if report["n_passed"] == report["n_designated"] else 2


def _demo(args):
    from .demo import DemoBackend
    from .pipeline import Pipeline
    path = Path(args.task).expanduser()
    if args.task in {"english", "history"}:
        path = Path(__file__).resolve().parents[2] / "examples" / (args.task + ".json")
        if not path.exists():
            raise ValueError("Use a source checkout for bundled examples, or pass --task /path/to/task.json")
    settings = Settings(backend="demo", schema_retries=0, max_repairs=1)
    result = Pipeline(DemoBackend(), settings, Path(args.output).expanduser()).run(_task(path))
    _print(result)
    return 0 if result["status"] == "accepted_by_checks" else 2


def _doctor(args):
    packages = {}
    for name in ("creativity-pipeline", "pydantic", "PyYAML", "torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": packages, "cuda": "not checked; use --check-cuda",
              "note": "Environment inspection only; this does not establish model memory fit."}
    if args.check_cuda:
        try:
            import torch
            report["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
            if torch.cuda.is_available():
                report["cuda"]["devices"] = [
                    {"index": index, "name": torch.cuda.get_device_name(index),
                     "vram_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2)}
                    for index in range(torch.cuda.device_count())
                ]
        except (ImportError, RuntimeError) as exc:
            report["cuda"] = {"error": str(exc)}
    _print(report)
    return 0


def _validate_data(args):
    from .training_data import validate_dataset
    report = validate_dataset(Path(args.path).expanduser())
    _print(report)
    return 0 if report["valid"] else 2


def _train(args):
    from .training import train
    report = train(Path(args.config).expanduser(), Path(args.output).expanduser(), dry_run=args.dry_run)
    _print(report)
    return 0


def _read_tasks(paths):
    tasks = []
    for value in paths:
        path = Path(value).expanduser()
        if path.suffix == ".jsonl":
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            raw = load_json(path)
            raw = raw if isinstance(raw, list) else [raw]
        tasks.extend(Task.model_validate(row) for row in raw)
    if not tasks:
        raise ValueError("Benchmark requires at least one task")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark task IDs must be unique")
    return tasks


def _benchmark(args):
    from .pipeline import Pipeline
    settings = _settings(args)
    tasks = _read_tasks(args.tasks)
    if len(args.strategies) != len(set(args.strategies)):
        raise ValueError("Benchmark strategies must be unique")
    root = Path(args.output).expanduser()
    root.mkdir(parents=True, exist_ok=False)
    backend = _backend(settings)
    results = []
    schedule = []
    by_task = {task.task_id: task for task in tasks}
    for task in tasks:
        for sample in range(args.samples):
            seed = settings.seed + sample * 1000
            for strategy in args.strategies:
                run_settings = settings.model_copy(update={"strategy": strategy, "seed": seed})
                schedule.append({"run_id": uuid4().hex, "task_id": task.task_id, "mode": task.mode,
                                 "strategy": strategy, "sample": sample, "seed": seed,
                                 "run": str(Path(task.task_id) / f"{strategy}_s{sample:03d}"),
                                 "configuration": run_settings.model_dump(), "status": "pending"})
    overview = {"note": "Same configured caps, not equal realized compute. Compare quality against measured tokens and time; all strategies retain the same draft review/repair loop.",
                "pipeline_version": __version__,
                "settings": settings.model_dump(), "samples_per_task_strategy": args.samples,
                "tasks": {task.task_id: task.model_dump() for task in tasks},
                "designated": schedule, "backend": backend.describe(), "results": results}
    atomic_json(root / "benchmark.json", overview)
    for entry in schedule:
        task = by_task[entry["task_id"]]
        destination = root / entry["run"]
        entry["status"] = "running"
        atomic_json(root / "benchmark.json", overview)
        try:
            summary = Pipeline(backend, Settings.model_validate(entry["configuration"]), destination, run_id=entry["run_id"]).run(task)
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            overview["backend"] = backend.describe()
            atomic_json(root / "benchmark.json", overview)
            raise
        entry["status"] = summary["status"]
        overview["backend"] = backend.describe()
        results.append({"run": entry["run"], **summary})
        atomic_json(root / "benchmark.json", overview)
        print(f"{task.task_id} {entry['strategy']} sample={entry['sample']}: {summary['status']} ({summary['tokens']['total_tokens']} tokens)", file=sys.stderr)
    _print({"output": str(root.resolve()), "runs": len(results),
            "accepted_by_checks": sum(row["status"] == "accepted_by_checks" for row in results),
            "note": overview["note"]})
    return 0 if all(row["status"] == "accepted_by_checks" for row in results) else 2


def _run_records(paths):
    from .evaluation import condition_identity
    records = []
    seen = set()

    def record(summary, configuration, task, text):
        if not summary.get("run_id"):
            raise ValueError("A run lacks a stable run_id. Legacy path-based identities cannot be exported; rerun it or supply explicit provenance records.")
        if summary.get("is_fixture") or summary.get("backend", {}).get("fixture"):
            raise ValueError("Refusing to export demo fixtures as model evaluation")
        task = Task.model_validate(task).model_dump()
        if summary["task_id"] != task["task_id"] or summary["mode"] != task["mode"]:
            raise ValueError("Run summary identity disagrees with the saved task")
        return {
            "output_id": f"{summary['task_id']}--{summary['run_id']}",
            "task_id": summary["task_id"], "mode": summary["mode"],
            "baseline": summary["strategy"], "origin": "model", "text": text,
            "status": summary["status"], "tokens": summary["tokens"],
            "seed": configuration["seed"], "backend": summary["backend"],
            "configuration": configuration,
            "condition_id": condition_identity(configuration, summary["backend"]),
            "task": task,
            "task_sha256": text_hash(json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        }

    for value in paths:
        directory = Path(value).expanduser().resolve()
        summaries = ([directory / "summary.json"] if (directory / "summary.json").exists()
                     else sorted(directory.rglob("summary.json")))
        benchmark_path = directory / "benchmark.json"
        if not summaries and not benchmark_path.exists():
            raise ValueError(f"No pipeline runs found in {directory}")
        for path in summaries:
            if path in seen:
                continue
            seen.add(path)
            summary = load_json(path)
            if summary.get("is_fixture") or summary.get("backend", {}).get("fixture"):
                raise ValueError(f"Refusing to export demo fixtures as model evaluation: {path.parent}")
            final = path.parent / "final.txt"
            text = final.read_bytes().decode("utf-8") if final.exists() else ""
            expected = summary.get("output_sha256")
            if expected is not None and text_hash(text) != expected:
                raise ValueError(f"Final text changed since the recorded run: {final}")
            if expected is None and text:
                raise ValueError(f"Unexpected final text without a recorded hash: {final}")
            configuration = load_json(path.parent / "config.json")
            configuration["pipeline_version"] = summary.get("pipeline_version", "unrecorded")
            records.append(record(summary, configuration, load_json(path.parent / "task.json"), text))
        if benchmark_path.exists():
            benchmark = load_json(benchmark_path)
            for entry in benchmark.get("designated", []):
                expected_summary = (directory / entry["run"] / "summary.json").resolve()
                if not expected_summary.is_relative_to(directory):
                    raise ValueError("Benchmark schedule contains a path outside its run directory")
                if expected_summary in seen or expected_summary.exists():
                    continue
                summary = {"run_id": entry["run_id"], "task_id": entry["task_id"], "mode": entry["mode"],
                           "strategy": entry["strategy"], "status": entry["status"],
                           "backend": benchmark.get("backend", {}),
                           "is_fixture": benchmark.get("settings", {}).get("backend") == "demo",
                           "tokens": {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                      "total_tokens": 0, "accounting": [], "complete": entry["status"] == "pending"}}
                configuration = dict(entry["configuration"])
                configuration["pipeline_version"] = benchmark.get("pipeline_version", "unrecorded")
                records.append(record(summary, configuration, benchmark["tasks"][entry["task_id"]], ""))
                seen.add(expected_summary)
    return records


def _collect_records(args):
    records = _run_records(args.runs) if args.runs else []
    if args.records:
        extra = load_json(Path(args.records).expanduser())
        if not isinstance(extra, list):
            raise ValueError("--records must contain a JSON list; declare origin explicitly for every row")
        if any(not isinstance(row, dict) for row in extra):
            raise ValueError("Each --records entry must be a JSON object")
        if any(row.get("is_fixture") or (isinstance(row.get("backend"), dict) and row["backend"].get("fixture")) for row in extra):
            raise ValueError("Fixture records cannot be used as a writing-quality evaluation")
        records.extend(extra)
    if not records:
        raise ValueError("Supply one or more run directories and/or --records with human controls")
    return records


def _freeze_outputs(args):
    from .evaluation import freeze_outputs
    records = freeze_outputs(_collect_records(args), Path(args.output).expanduser())
    _print({"manifest": str(Path(args.output).expanduser().resolve()), "designated_outputs": len(records)})
    return 0


def _export_detectors(args):
    from .evaluation import freeze_outputs, write_detector_csv_template
    csv_path = Path(args.csv).expanduser()
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite detector sheet: {csv_path}")
    records = _collect_records(args)
    manifest = freeze_outputs(records, Path(args.manifest).expanduser())
    write_detector_csv_template(manifest, csv_path, detector=args.detector or "", version=args.version or "")
    _print({"designated_outputs": len(manifest), "manifest": str(Path(args.manifest).resolve()),
            "csv": str(csv_path.resolve()), "note": "No detector service was queried. Keep the frozen texts unchanged and fill in the manual sheet."})
    return 0


def _export_review(args):
    from .human_evaluation import export_blind_review
    report = export_blind_review(Path(args.manifest).expanduser(), Path(args.output).expanduser(), seed=args.seed)
    _print(report)
    return 0


def _import_review(args):
    from .human_evaluation import import_human_ratings
    path = Path(args.output).expanduser()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a report: {path}")
    report = import_human_ratings(Path(args.csv).expanduser(), Path(args.key).expanduser())
    atomic_json(path, report)
    _print(report)
    return 0


def _import_detectors(args):
    from .evaluation import aggregate_detector_results, import_detector_csv
    path = Path(args.output).expanduser()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite a report: {path}")
    manifest = Path(args.manifest).expanduser()
    rows = import_detector_csv(Path(args.csv).expanduser(), manifest)
    report = aggregate_detector_results(rows, manifest)
    atomic_json(path, report)
    _print(report)
    return 0


def _export_training(args):
    directory = Path(args.run).expanduser()
    summary = load_json(directory / "summary.json")
    if summary.get("is_fixture") or summary.get("backend", {}).get("fixture"):
        raise ValueError("Demo fixtures are not a training corpus")
    requests = {}
    records = []
    skipped_thinking = 0
    for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["kind"] == "request":
            requests[event["call_id"]] = event["request"]
        elif event["kind"] == "response":
            request = requests.get(event["call_id"])
            response = event["response"]
            if not request or request["role"] not in args.roles or response["finish_reason"] != "stop":
                continue
            if request.get("thinking"):
                skipped_thinking += 1
                continue
            if response.get("metadata", {}).get("complete") is False:
                continue
            try:
                contract = Plan if request["role"] == "plan" else Draft
                contract.model_validate(parse_json_object(response["text"]))
            except ValueError:
                continue
            records.append({
                "id": f"{args.group_id}-{summary.get('run_id', text_hash(str(directory.resolve()))[:16])}-{event['call_id']}",
                "group_id": args.group_id, "approach_id": args.approach_id,
                "mode": summary["mode"], "split": args.split,
                "messages": request["messages"], "completion": response["text"],
                "accepted": False,
                "provenance": f"UNCURATED pipeline proposal; task={summary['task_id']}; role={request['role']}; completion_sha256={text_hash(response['text'])}. Record human review and source rights before accepting.",
            })
    if not records:
        raise ValueError("No complete, schema-valid proposals for the requested roles")
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    _print({"output": str(path.resolve()), "rows": len(records), "accepted_rows": 0, "skipped_thinking_requests": skipped_thinking,
            "note": "Review/correct each proposal, assign its real approach_id and provenance, and explicitly accept useful examples. These rows are currently excluded from training."})
    return 0


def _inference_arguments(parser):
    parser.add_argument("--config", help="YAML settings; defaults to built-in local settings")
    parser.add_argument("--allow-download", action="store_true", help="Allow model/tokenizer downloads instead of requiring cached files")
    parser.add_argument("--adapter", help="Optional trained PEFT adapter directory")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--critic-model", dest="critic_model_id", help="Optional separate critic checkpoint; models run sequentially")
    parser.add_argument("--critic-revision", help="Revision for the optional separate critic")


def _source_arguments(parser):
    parser.add_argument("--source-library", help="Local SQLite source library built by build-sources")
    parser.add_argument("--source-query", help="Retrieval query; defaults to the task prompt")
    parser.add_argument("--counter-query", action="append", default=[], help="Additional query for competing accounts; repeatable")
    parser.add_argument("--source-max-chars", type=int, default=8000, help="Retrieved source JSON character budget; final token check still applies")
    parser.add_argument("--source-max-passages", type=int, default=6)
    parser.add_argument("--source-per-document", type=int, default=2)
    parser.add_argument("--source-min-documents", type=int, default=3, help="Diversity target, never filled with irrelevant documents")


def parser():
    root = argparse.ArgumentParser(prog="creativity", description="Local, bounded creative and historical writing experiments")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Offline deterministic plumbing test; not model writing")
    demo.add_argument("--task", default="english", help="english, history, or task JSON path")
    demo.add_argument("--output", required=True)
    demo.set_defaults(func=_demo)
    generate = sub.add_parser("generate", help="Generate, check, and revise one task")
    generate.add_argument("task")
    generate.add_argument("--output", required=True)
    generate.add_argument("--state-in", help="Previous accepted English state.json")
    generate.add_argument("--strategy", choices=("direct", "single_plan", "multi_plan"))
    _inference_arguments(generate)
    _source_arguments(generate)
    generate.set_defaults(func=_generate)
    critique = sub.add_parser("critique", help="Assess a supplied draft without generating or rewriting it")
    critique.add_argument("task")
    critique.add_argument("--draft", required=True, help="UTF-8 plain text draft")
    critique.add_argument("--output", required=True)
    _inference_arguments(critique)
    _source_arguments(critique)
    critique.set_defaults(func=_critique)
    critic_eval = sub.add_parser("evaluate-critic", help="Assess a critic on labeled development cases without generating drafts")
    critic_eval.add_argument("--cases", required=True, help="JSONL of task, supplied draft, and explicit expected judgments")
    critic_eval.add_argument("--output", required=True)
    _inference_arguments(critic_eval)
    critic_eval.set_defaults(func=_evaluate_critic)
    build = sub.add_parser("build-sources", help="Create an immutable, local source-passage library from JSONL")
    build.add_argument("path")
    build.add_argument("--output", required=True)
    build.add_argument("--chunk-chars", type=int, default=1800)
    build.add_argument("--overlap-chars", type=int, default=200)
    build.set_defaults(func=_build_sources)
    retrieve = sub.add_parser("retrieve-sources", help="Select passages and record all library omissions")
    retrieve.add_argument("library")
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--counter-query", action="append", default=[])
    retrieve.add_argument("--output", required=True)
    retrieve.add_argument("--max-chars", type=int, default=8000)
    retrieve.add_argument("--max-passages", type=int, default=6)
    retrieve.add_argument("--per-document", type=int, default=2)
    retrieve.add_argument("--min-documents", type=int, default=3)
    retrieve.set_defaults(func=_retrieve_sources)
    quote = sub.add_parser("verify-source-quote", help="Check a quotation against its exact stored passage and original")
    quote.add_argument("library")
    quote.add_argument("--source-id", required=True)
    quote.add_argument("--quote", required=True)
    quote.set_defaults(func=_verify_source_quote)
    doctor = sub.add_parser("doctor", help="Inspect dependencies without downloading weights")
    doctor.add_argument("--check-cuda", action="store_true")
    doctor.set_defaults(func=_doctor)
    validate = sub.add_parser("validate-data", help="Validate curated SFT JSONL and split isolation")
    validate.add_argument("path")
    validate.set_defaults(func=_validate_data)
    training = sub.add_parser("train", help="Train a QLoRA adapter or validate its inputs")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--dry-run", action="store_true")
    training.set_defaults(func=_train)
    benchmark = sub.add_parser("benchmark", help="Run sequential plan ablations with measured compute")
    benchmark.add_argument("--tasks", nargs="+", required=True, help="Task JSON files, JSON lists, or JSONL")
    benchmark.add_argument("--strategies", nargs="+", choices=("direct", "single_plan", "multi_plan"), default=["direct", "single_plan", "multi_plan"])
    benchmark.add_argument("--samples", type=int, choices=range(1, 21), default=1, metavar="1..20")
    benchmark.add_argument("--output", required=True)
    _inference_arguments(benchmark)
    benchmark.set_defaults(func=_benchmark)
    freeze = sub.add_parser("freeze-outputs", help="Freeze all designated outputs before external evaluation")
    freeze.add_argument("runs", nargs="*")
    freeze.add_argument("--records", help="Additional explicit output records, such as known-human controls")
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(func=_freeze_outputs)
    review = sub.add_parser("export-review", help="Create a blinded human-review packet from frozen outputs")
    review.add_argument("--manifest", required=True)
    review.add_argument("--output", required=True)
    review.add_argument("--seed", type=int, default=42)
    review.set_defaults(func=_export_review)
    rated = sub.add_parser("import-review", help="Validate blinded ratings and report separate quality dimensions")
    rated.add_argument("csv")
    rated.add_argument("--key", required=True, help="The packet's private_key.json; do not send this to reviewers")
    rated.add_argument("--output", required=True)
    rated.set_defaults(func=_import_review)
    export = sub.add_parser("export-detectors", help="Freeze run outputs and create a manual detector sheet")
    export.add_argument("runs", nargs="*")
    export.add_argument("--records", help="Additional JSON records, e.g. known-human controls")
    export.add_argument("--manifest", required=True)
    export.add_argument("--csv", required=True)
    export.add_argument("--detector")
    export.add_argument("--version")
    export.set_defaults(func=_export_detectors)
    ingest = sub.add_parser("import-detectors", help="Validate detector observations and aggregate by domain/version")
    ingest.add_argument("csv")
    ingest.add_argument("--manifest", required=True)
    ingest.add_argument("--output", required=True)
    ingest.set_defaults(func=_import_detectors)
    export_train = sub.add_parser("export-training", help="Export unaccepted proposals for human curation")
    export_train.add_argument("run")
    export_train.add_argument("--group-id", required=True, help="Keep all related source material/prompts in this same split group")
    export_train.add_argument("--approach-id", default="assign_after_review")
    export_train.add_argument("--split", choices=("train", "validation", "test"), required=True)
    export_train.add_argument("--roles", nargs="+", choices=("plan", "draft", "repair"), default=["plan", "repair"])
    export_train.add_argument("--output", required=True)
    export_train.set_defaults(func=_export_training)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("creativity: interrupted; completed run records and the designated benchmark schedule were retained", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"creativity: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
