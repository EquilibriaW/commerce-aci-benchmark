#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI helpers for pack evaluation, fuzzing, and guidance."""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Any

from pack_api.loader import PackRegistry
from pack_api.runtime import (
    build_fuzz_cases,
    eval_results_to_dicts,
    gate_should_fail,
    run_evaluators_on_trace,
    suggest_guidance_patch,
    write_eval_results,
)

import tomllib


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _print_results(results: list[dict[str, Any]]) -> None:
    for res in results:
        status = str(res.get("decision") or "uncertain").upper()
        print(f"{res.get('pack_id')}:{res.get('evaluator_id')} [{res.get('severity')}] {status} - {res.get('summary')}")


def _load_policy(policy_path: str | None) -> dict[str, Any]:
    if not policy_path:
        return {}
    data = tomllib.loads(Path(policy_path).read_text(encoding="utf-8"))
    return dict(data.get("policy") or data)


def _load_prompt(prompt: str | None, prompt_file: str | None) -> str | None:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt:
        return prompt
    return None


def _record_judge_prompt(trace: dict[str, Any], prompt: str | None, label: str = "global") -> bool:
    if not prompt:
        return False
    meta = trace.get("meta") or {}
    prompts = dict(meta.get("judge_system_prompts") or {})
    hashes = dict(meta.get("judge_system_prompt_hashes") or {})
    prompts[label] = prompt
    hashes[label] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    meta["judge_system_prompts"] = prompts
    meta["judge_system_prompt_hashes"] = hashes
    trace["meta"] = meta
    return True


def cmd_list_packs(args: argparse.Namespace) -> None:
    registry = PackRegistry.discover()
    for pack in registry.list_packs():
        print(f"{pack.id}\t{pack.name}\t{pack.version}\t{pack.description}")


def cmd_list_components(args: argparse.Namespace) -> None:
    registry = PackRegistry.discover()
    kind = args.type
    pack_filter = set(_parse_csv(args.pack))
    for pack in registry.list_packs():
        if pack_filter and pack.id not in pack_filter:
            continue
        if kind in ("evaluators", "all"):
            for spec in pack.evaluators:
                print(f"evaluator\t{pack.id}:{spec.id}\t{spec.kind}\t{spec.severity}\t{spec.entrypoint}\t{spec.description}")
        if kind in ("fuzzers", "all"):
            for spec in pack.fuzzers:
                print(f"fuzzer\t{pack.id}:{spec.id}\t{spec.kind}\t{spec.entrypoint}\t{spec.description}")
        if kind in ("guidance", "all"):
            for spec in pack.guidance:
                entry = spec.entrypoint or ""
                print(f"guidance\t{pack.id}:{spec.id}\t{spec.kind}\t{entry}\t{spec.description}")
        if kind in ("advisors", "all"):
            for spec in pack.advisors:
                print(f"advisor\t{pack.id}:{spec.id}\t{spec.kind}\t{spec.entrypoint}\t{spec.description}")


def cmd_eval_trace(args: argparse.Namespace) -> None:
    trace_path = Path(args.trace)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    selectors = _parse_csv(args.eval_packs)
    registry = PackRegistry.discover()
    policy = _load_policy(args.policy)
    judge_prompt = _load_prompt(args.judge_system_prompt, args.judge_system_prompt_file)
    if _record_judge_prompt(trace, judge_prompt):
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    results = run_evaluators_on_trace(
        trace,
        selected_evaluators=selectors,
        registry=registry,
        judge_system_prompt=judge_prompt,
    )
    out_path = write_eval_results(trace_path, results, policy=policy)
    results_dicts = eval_results_to_dicts(results)

    _print_results(results_dicts)
    print(f"gate_failed={gate_should_fail(results, policy)}")
    print(f"eval_results={out_path}")

    if args.json:
        print(json.dumps({"trace": str(trace_path), "results": results_dicts}, indent=2))
    if gate_should_fail(results, policy):
        sys.exit(2)


def cmd_eval_dir(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    selectors = _parse_csv(args.eval_packs)
    policy = _load_policy(args.policy)
    registry = PackRegistry.discover()
    traces = sorted(root.rglob("trace.json"))
    summary = []
    gate_failed_any = False

    judge_prompt = _load_prompt(args.judge_system_prompt, args.judge_system_prompt_file)
    for trace_path in traces:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if _record_judge_prompt(trace, judge_prompt):
            trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        results = run_evaluators_on_trace(
            trace,
            selected_evaluators=selectors,
            registry=registry,
            judge_system_prompt=judge_prompt,
        )
        out_path = write_eval_results(trace_path, results, policy=policy)
        gate_failed = gate_should_fail(results, policy)
        gate_failed_any = gate_failed_any or gate_failed
        summary.append({
            "trace": str(trace_path),
            "eval_results": str(out_path),
            "gate_failed": gate_failed,
            "results": eval_results_to_dicts(results),
        })
        print(f"{trace_path} -> {out_path}")

    if args.json:
        print(json.dumps({"traces": summary}, indent=2))
    if gate_failed_any:
        sys.exit(2)


def cmd_fuzz_generate(args: argparse.Namespace) -> None:
    registry = PackRegistry.discover()
    fuzzer_spec = registry.get_fuzzer_spec(args.fuzz_pack, args.fuzzer)
    fuzzer_fn = registry.load_fuzzer(args.fuzz_pack, args.fuzzer)
    turns = [int(x.strip()) for x in args.turns.split(",") if x.strip()]
    cases = build_fuzz_cases(fuzzer_fn, turns, config=fuzzer_spec.default_config)
    payload = [case.__dict__ for case in cases]
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


def cmd_guidance_suggest(args: argparse.Namespace) -> None:
    registry = PackRegistry.discover()
    advisor_spec = registry.get_advisor_spec(args.advisor_pack, args.advisor)
    advisor_fn = registry.load_advisor(args.advisor_pack, args.advisor)
    trace_path = Path(args.trace)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    eval_results = []
    eval_path = trace_path.parent / "eval_results.json"
    if eval_path.exists():
        try:
            eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
            eval_results = eval_payload.get("results") or []
        except Exception:
            eval_results = []

    patch = suggest_guidance_patch(advisor_fn, trace, eval_results, config=advisor_spec.default_config)
    if patch is None:
        print("skipped: advisor returned no patch")
        return
    out_path = Path(args.out)
    out_path.write_text(json.dumps(patch.__dict__, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_packs = sub.add_parser("list-packs", help="List discovered packs")
    list_packs.set_defaults(func=cmd_list_packs)

    list_components = sub.add_parser("list-components", help="List pack components")
    list_components.add_argument("--type", choices=["evaluators", "fuzzers", "guidance", "advisors", "all"], default="all")
    list_components.add_argument("--pack", type=str, default="")
    list_components.set_defaults(func=cmd_list_components)

    eval_trace = sub.add_parser("eval-trace", help="Run evaluators on a trace")
    eval_trace.add_argument("--trace", type=str, required=True)
    eval_trace.add_argument(
        "--eval-packs",
        "--packs",
        dest="eval_packs",
        type=str,
        default="",
        help="Comma-separated selectors: pack_id or pack_id:evaluator_id (use fully-qualified IDs when in doubt)",
    )
    eval_trace.add_argument("--policy", type=str, default="")
    eval_trace.add_argument("--judge-system-prompt", type=str, default="", help="System prompt for LLM evaluators")
    eval_trace.add_argument("--judge-system-prompt-file", type=str, default="", help="File containing system prompt for LLM evaluators")
    eval_trace.add_argument("--json", action="store_true")
    eval_trace.set_defaults(func=cmd_eval_trace)

    eval_dir = sub.add_parser("eval-dir", help="Run evaluators on all traces in a directory")
    eval_dir.add_argument("--dir", type=str, required=True)
    eval_dir.add_argument(
        "--eval-packs",
        "--packs",
        dest="eval_packs",
        type=str,
        default="",
        help="Comma-separated selectors: pack_id or pack_id:evaluator_id (use fully-qualified IDs when in doubt)",
    )
    eval_dir.add_argument("--policy", type=str, default="")
    eval_dir.add_argument("--judge-system-prompt", type=str, default="", help="System prompt for LLM evaluators")
    eval_dir.add_argument("--judge-system-prompt-file", type=str, default="", help="File containing system prompt for LLM evaluators")
    eval_dir.add_argument("--json", action="store_true")
    eval_dir.set_defaults(func=cmd_eval_dir)

    fuzz_generate = sub.add_parser("fuzz-generate", help="Generate fuzz cases from a pack")
    fuzz_generate.add_argument("--fuzz-pack", type=str, required=True)
    fuzz_generate.add_argument("--fuzzer", type=str, required=True)
    fuzz_generate.add_argument("--turns", type=str, required=True, help="Comma-separated injection turns")
    fuzz_generate.add_argument("--out", type=str, required=True)
    fuzz_generate.set_defaults(func=cmd_fuzz_generate)

    guidance_suggest = sub.add_parser("guidance-suggest", help="Suggest a guidance patch from a trace")
    guidance_suggest.add_argument("--trace", type=str, required=True)
    guidance_suggest.add_argument("--advisor-pack", type=str, required=True)
    guidance_suggest.add_argument("--advisor", type=str, required=True)
    guidance_suggest.add_argument("--out", type=str, required=True)
    guidance_suggest.set_defaults(func=cmd_guidance_suggest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
