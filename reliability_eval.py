#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run reliability sweeps across seeded variants and summarize outcomes."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from benchmark_computeruse import (
    ComputerUseAgent,
    SYSTEM_PROMPT,
    TASKS,
    URL_BASELINE,
    URL_TREATMENT,
    URL_TREATMENT_DOCS,
    run_trial,
)


APP_URLS = {
    "baseline": URL_BASELINE,
    "treatment": URL_TREATMENT,
    "treatment-docs": URL_TREATMENT_DOCS,
}


def parse_seeds(spec: str) -> list[int]:
    if not spec:
        raise ValueError("Seeds spec is empty.")
    seeds: list[int] = []
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            for seed in range(start, end + step, step):
                seeds.append(seed)
        else:
            seeds.append(int(part))
    return seeds


def aggregate_event_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for run in runs:
        counts = run.get("event_counts_by_type") or {}
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def failure_stage_histogram(runs: list[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for run in runs:
        if run.get("success"):
            continue
        stage = run.get("stage") or "unknown"
        hist[stage] = hist.get(stage, 0) + 1
    return hist


def failure_reason_histogram(runs: list[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for run in runs:
        if run.get("success"):
            continue
        reason = run.get("failure_reason") or "unknown"
        hist[reason] = hist.get(reason, 0) + 1
    return hist


def read_prompt_override(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


async def run_single(
    *,
    task: dict[str, Any],
    target_url: str,
    api_url: str,
    condition_name: str,
    run_num: int,
    discoverability: str,
    capability: str,
    variant_seed: int,
    variant_level: int,
    model: str,
    system_prompt: str,
    is_adaptive: bool = False,
    base_seed: Optional[int] = None,
) -> dict[str, Any]:
    try:
        result = await run_trial(
            task,
            target_url,
            api_url,
            condition_name,
            run_num,
            discoverability=discoverability,
            capability=capability,
            variant_seed=variant_seed,
            variant_level=variant_level,
            system_prompt=system_prompt,
            model=model,
        )

        final_state = result.get("final_state") or {}
        events = final_state.get("events") or {}
        event_counts = events.get("counts_by_type") or {}
        last_events = events.get("last_n") or []
        last_event_type = last_events[-1].get("type") if last_events else None
        action_log = result.get("action_log") or []
        action_log_tail = action_log[-5:] if action_log else []
        progress = final_state.get("progress") or {}
        cart = final_state.get("cart") or {}
        cart_total_items = cart.get("total_items", 0)
        cart_items = cart.get("items") or []
        cart_items_preview = [
            {
                "slug": item.get("slug"),
                "variant": item.get("variant"),
                "quantity": item.get("quantity"),
            }
            for item in cart_items
        ][:3]
        stage = final_state.get("stage") or ("error" if final_state.get("error") else "unknown")
        success = bool(result.get("success", False))
        if success:
            failure_reason = "success"
        elif stage == "browse":
            failure_reason = "no_items_added"
        elif stage == "cart":
            failure_reason = "checkout_not_started"
        elif stage == "checkout":
            failure_reason = "checkout_incomplete"
        elif stage == "done":
            failure_reason = "verifier_failed"
        elif stage == "error":
            failure_reason = "state_error"
        else:
            failure_reason = "unknown"

        missing_events: list[str] = []
        if not progress.get("has_cart_items"):
            missing_events.append("ADD_TO_CART")
        if not progress.get("checkout_started"):
            missing_events.append("START_CHECKOUT")
        if not progress.get("order_completed"):
            missing_events.append("CHECKOUT_COMPLETE")
        run_record = {
            "seed": variant_seed,
            "variant_level": variant_level,
            "success": success,
            "steps": result.get("steps", 0),
            "model_calls": result.get("model_calls", 0),
            "ui_actions": result.get("ui_actions", 0),
            "wall_time_seconds": result.get("wall_time_seconds", 0.0),
            "stage": stage,
            "events_total": events.get("total", 0),
            "event_counts_by_type": event_counts,
            "last_event_type": last_event_type,
            "action_log_tail": action_log_tail,
            "missing_events": missing_events,
            "failure_reason": failure_reason,
            "cart_total_items": cart_total_items,
            "cart_items_preview": cart_items_preview,
            "trace_path": result.get("trace_path"),
        }
    except Exception as exc:
        run_record = {
            "seed": variant_seed,
            "variant_level": variant_level,
            "success": False,
            "steps": 0,
            "model_calls": 0,
            "ui_actions": 0,
            "wall_time_seconds": 0.0,
            "stage": "error",
            "events_total": 0,
            "event_counts_by_type": {},
            "trace_path": None,
            "error": str(exc),
        }
    if is_adaptive:
        run_record["is_adaptive"] = True
        run_record["base_seed"] = base_seed if base_seed is not None else variant_seed
    return run_record


async def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    task = next((t for t in TASKS if t["id"] == args.task), None)
    if not task:
        raise ValueError(f"Unknown task '{args.task}'. Available: {[t['id'] for t in TASKS]}")

    base_url = APP_URLS[args.app]
    target_url = base_url if args.start == "root" else f"{base_url}/agent"
    condition_name = f"Reliability/{args.app}/{args.start}/{args.discoverability}/{args.capability}"

    system_prompt = read_prompt_override(args.prompt_file) or SYSTEM_PROMPT
    seeds = parse_seeds(args.seeds)
    base_level = max(0, min(3, int(args.variant_level)))

    run_counter = 1
    base_runs: list[dict[str, Any]] = []
    for seed in seeds:
        run_record = await run_single(
            task=task,
            target_url=target_url,
            api_url=base_url,
            condition_name=condition_name,
            run_num=run_counter,
            discoverability=args.discoverability,
            capability=args.capability,
            variant_seed=seed,
            variant_level=base_level,
            model=args.model,
            system_prompt=system_prompt,
        )
        base_runs.append(run_record)
        run_counter += 1

    summary = {
        "success_rate": sum(1 for r in base_runs if r.get("success")) / max(1, len(base_runs)),
        "failure_stage_histogram": failure_stage_histogram(base_runs),
        "failure_reason_histogram": failure_reason_histogram(base_runs),
        "event_coverage": {
            "unique_event_types_seen": len(aggregate_event_counts(base_runs)),
            "counts_by_type": aggregate_event_counts(base_runs),
        },
    }

    adaptive_runs: list[dict[str, Any]] = []
    if args.adaptive:
        budget = args.adaptive_budget
        for run in base_runs:
            if budget <= 0:
                break
            if run.get("success"):
                continue
            new_level = min(base_level + 1, 3)
            if new_level == run.get("variant_level"):
                continue
            adaptive_run = await run_single(
                task=task,
                target_url=target_url,
                api_url=base_url,
                condition_name=condition_name,
                run_num=run_counter,
                discoverability=args.discoverability,
                capability=args.capability,
                variant_seed=int(run.get("seed", 0)),
                variant_level=new_level,
                model=args.model,
                system_prompt=system_prompt,
                is_adaptive=True,
                base_seed=int(run.get("seed", 0)),
            )
            adaptive_runs.append(adaptive_run)
            run_counter += 1
            budget -= 1

    expanded_summary = {}
    if adaptive_runs:
        expanded_runs = base_runs + adaptive_runs
        expanded_summary = {
            "base_success_rate": summary["success_rate"],
            "expanded_success_rate": sum(1 for r in expanded_runs if r.get("success")) / max(1, len(expanded_runs)),
            "new_failures_found": sum(1 for r in adaptive_runs if not r.get("success")),
            "failure_stage_shift": {
                "base": summary["failure_stage_histogram"],
                "adaptive": failure_stage_histogram(adaptive_runs),
            },
            "failure_reason_shift": {
                "base": summary["failure_reason_histogram"],
                "adaptive": failure_reason_histogram(adaptive_runs),
            },
            "event_coverage": {
                "unique_event_types_seen": len(aggregate_event_counts(expanded_runs)),
                "counts_by_type": aggregate_event_counts(expanded_runs),
            },
        }

    model_full_name = ComputerUseAgent.SUPPORTED_MODELS.get(args.model, args.model)
    report = {
        "meta": {
            "app": args.app,
            "task": task["id"],
            "task_instruction": task.get("instruction"),
            "start": args.start,
            "discoverability": args.discoverability,
            "capability": args.capability,
            "model": model_full_name,
            "model_alias": args.model,
            "seeds": args.seeds,
            "variant_level": base_level,
            "adaptive": args.adaptive,
            "adaptive_budget": args.adaptive_budget,
            "prompt_file": args.prompt_file,
            "timestamp": datetime.now().isoformat(),
        },
        "base_runs": base_runs,
        "adaptive_runs": adaptive_runs,
        "summary": summary,
        "expanded_summary": expanded_summary,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reliability evaluation across seeded variants.")
    parser.add_argument("--app", choices=["baseline", "treatment", "treatment-docs"], required=True)
    parser.add_argument("--task", required=True, help="Task id (one of TASKS in benchmark_computeruse.py)")
    parser.add_argument("--start", choices=["root", "agent"], default="root")
    parser.add_argument("--discoverability", choices=["navbar", "hidden"], default="navbar")
    parser.add_argument("--capability", choices=["advantage", "parity"], default="advantage")
    parser.add_argument("--model", choices=["sonnet", "haiku"], default=ComputerUseAgent.DEFAULT_MODEL)
    parser.add_argument("--seeds", default="0-9", help="Seed list like '0-9' or '0,1,2'")
    parser.add_argument(
        "--variant-level",
        type=int,
        default=0,
        help="Base variant level (0-3). Higher levels add stronger UI perturbations.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Rerun failing seeds at level+1 (capped at 3) until budget is exhausted.",
    )
    parser.add_argument(
        "--adaptive-budget",
        type=int,
        default=10,
        help="Maximum number of adaptive reruns.",
    )
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_path = args.out or f"benchmark_results/reliability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = asyncio.run(run_sweep(args))

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote reliability report to {output_path}")


if __name__ == "__main__":
    main()
