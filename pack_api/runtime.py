#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack runtime helpers for evaluators, fuzzers, guidance, and advisors."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from pack_api.contracts import EvalResult, FuzzCase, GuidancePatch
from pack_api.loader import PackRegistry
from trace_schema import validate_trace_v2


def _normalize_severity(value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"warning", "warn"}:
        return "warn"
    if value in {"error", "err"}:
        return "error"
    return value or "warn"


def _normalize_decision(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    value = str(value or "").strip().lower()
    if value in {"pass", "fail", "uncertain"}:
        return value
    if value in {"passed", "ok", "success"}:
        return "pass"
    if value in {"failed", "error"}:
        return "fail"
    return "uncertain"


def _ensure_eval_result(
    result: EvalResult | dict[str, Any],
    pack_id: str,
    evaluator_id: str,
    default_severity: str,
) -> EvalResult:
    if isinstance(result, EvalResult):
        eval_result = result
    elif isinstance(result, dict):
        decision = result.get("decision")
        if decision is None and "passed" in result:
            decision = "pass" if bool(result.get("passed")) else "fail"
        eval_result = EvalResult(
            pack_id=result.get("pack_id") or pack_id,
            evaluator_id=result.get("evaluator_id") or evaluator_id,
            decision=_normalize_decision(decision),
            severity=_normalize_severity(str(result.get("severity") or default_severity)),
            confidence=result.get("confidence"),
            metrics=dict(result.get("metrics") or {}),
            summary=str(result.get("summary") or ""),
            evidence=list(result.get("evidence") or []),
        )
    else:
        raise TypeError("Evaluator must return EvalResult or dict")

    if not eval_result.pack_id:
        eval_result.pack_id = pack_id
    if not eval_result.evaluator_id:
        eval_result.evaluator_id = evaluator_id
    if not eval_result.severity:
        eval_result.severity = _normalize_severity(default_severity)
    if not eval_result.decision:
        eval_result.decision = "uncertain"
    return eval_result


def _merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults or {})
    merged.update(overrides or {})
    return merged


def _select_override(
    config_overrides: Optional[dict[str, dict[str, Any]]],
    pack_id: str,
    evaluator_id: str,
) -> dict[str, Any]:
    if not config_overrides:
        return {}
    if f"{pack_id}:{evaluator_id}" in config_overrides:
        return config_overrides[f"{pack_id}:{evaluator_id}"]
    if evaluator_id in config_overrides:
        return config_overrides[evaluator_id]
    if pack_id in config_overrides:
        return config_overrides[pack_id].get(evaluator_id, {})
    return {}


def _resolve_pack_ids(
    registry: PackRegistry,
    selected_packs: Optional[list[str]],
) -> list[str]:
    if not selected_packs:
        return [p.id for p in registry.list_packs()]
    return [p for p in selected_packs if p]


def _resolve_component_selectors(
    registry: PackRegistry,
    selected_packs: Optional[list[str]],
    selected_items: Optional[list[str]],
    component_attr: str,
) -> list[tuple[str, str]]:
    """Resolve component selectors into concrete (pack_id, component_id) pairs."""
    resolved: list[tuple[str, str]] = []
    seen = set()

    def add_pair(pack_id: str, item_id: str) -> None:
        key = (pack_id, item_id)
        if key in seen:
            return
        seen.add(key)
        resolved.append(key)

    pack_ids = _resolve_pack_ids(registry, selected_packs)
    manifests = {p.id: p for p in registry.list_packs()}
    for pack_id in pack_ids:
        if pack_id not in manifests:
            raise ValueError(f"Pack not found: {pack_id}")

    component_ids_by_pack: dict[str, set[str]] = {}
    component_index: dict[str, list[str]] = {}
    for pack_id in pack_ids:
        manifest = manifests[pack_id]
        ids = {spec.id for spec in getattr(manifest, component_attr) if spec.id}
        component_ids_by_pack[pack_id] = ids
        for cid in ids:
            component_index.setdefault(cid, []).append(pack_id)

    if not selected_items:
        for pack_id in pack_ids:
            manifest = manifests[pack_id]
            for spec in getattr(manifest, component_attr):
                add_pair(pack_id, spec.id)
        return resolved

    for item in selected_items:
        if not item:
            continue
        if ":" in item:
            pack_id, item_id = item.split(":", 1)
            if pack_id not in pack_ids:
                raise ValueError(f"Pack not found for selector: {item}")
            if item_id not in component_ids_by_pack.get(pack_id, set()):
                raise ValueError(f"Component not found: {item}")
            add_pair(pack_id, item_id)
            continue
        if item in pack_ids:
            manifest = manifests[item]
            for spec in getattr(manifest, component_attr):
                add_pair(item, spec.id)
            continue
        matches = component_index.get(item, [])
        if not matches:
            raise ValueError(f"Component id not found: {item}")
        if len(matches) > 1:
            raise ValueError(
                "Component id is ambiguous across packs. "
                f"Use pack_id:{item}. Matches: {', '.join(sorted(matches))}"
            )
        add_pair(matches[0], item)

    return resolved


def assemble_system_prompt(base_prompt: str, guidance_fragments: list[str]) -> str:
    if not guidance_fragments:
        return base_prompt
    lines = [base_prompt.rstrip(), "", "### Guidance"]
    for frag in guidance_fragments:
        frag = str(frag).strip()
        if frag:
            lines.append(f"- {frag}")
    return "\n".join(lines).strip()


def load_guidance_fragments(
    registry: PackRegistry,
    selected_guidance: Optional[list[str]] = None,
    overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> list[str]:
    fragments: list[str] = []
    if not selected_guidance:
        return fragments
    selectors = _resolve_component_selectors(registry, None, selected_guidance, "guidance")
    for pack_id, guidance_id in selectors:
        spec = registry.get_guidance_spec(pack_id, guidance_id)
        config = _merge_config(spec.default_config, _select_override(overrides, pack_id, guidance_id))
        if spec.kind == "static":
            fragments.extend([str(f) for f in (spec.text_fragments or []) if str(f).strip()])
            continue
        if not spec.entrypoint:
            continue
        fn = registry.load_guidance(pack_id, guidance_id)
        try:
            result = fn(config)
        except TypeError:
            result = fn()
        if isinstance(result, str):
            fragments.append(result)
        elif isinstance(result, list):
            fragments.extend([str(f) for f in result if str(f).strip()])
        elif result is not None:
            fragments.append(str(result))
    return fragments


def run_evaluators_on_trace(
    trace: dict[str, Any],
    selected_packs: Optional[list[str]] = None,
    selected_evaluators: Optional[list[str]] = None,
    config_overrides: Optional[dict[str, dict[str, Any]]] = None,
    judge_system_prompt: Optional[str] = None,
    registry: Optional[PackRegistry] = None,
) -> list[EvalResult]:
    """Run pack evaluators on a trace and return results."""
    validate_trace_v2(trace)
    registry = registry or PackRegistry.discover()

    selectors = _resolve_component_selectors(registry, selected_packs, selected_evaluators, "evaluators")
    results: list[EvalResult] = []

    for pack_id, evaluator_id in selectors:
        spec = registry.get_evaluator_spec(pack_id, evaluator_id)
        overrides = _select_override(config_overrides, pack_id, evaluator_id)
        config = _merge_config(spec.default_config, overrides)
        if spec.kind == "llm" and judge_system_prompt and not config.get("system_prompt"):
            config["system_prompt"] = judge_system_prompt
        try:
            evaluator = registry.load_evaluator(pack_id, evaluator_id)
            raw_result = evaluator(trace, config)
            eval_result = _ensure_eval_result(raw_result, pack_id, evaluator_id, spec.severity)
        except Exception as exc:
            tb_lines = traceback.format_exc().strip().splitlines()
            excerpt = "\n".join(tb_lines[-8:]) if tb_lines else str(exc)
            eval_result = EvalResult(
                pack_id=pack_id,
                evaluator_id=evaluator_id,
                decision="fail",
                severity="error",
                confidence=None,
                summary=f"exception: {exc}",
                evidence=[{"note": excerpt}],
                metrics={},
            )
        results.append(eval_result)
    return results


def gate_decision(results: Iterable[EvalResult], policy: Optional[dict[str, Any]] = None) -> bool:
    policy = dict(policy or {})
    block_on_uncertain = bool(policy.get("block_on_uncertain", False))
    for res in results:
        if res.severity == "error" and res.decision == "fail":
            return False
    if block_on_uncertain:
        for res in results:
            if res.decision == "uncertain":
                return False
    return True


def gate_should_fail(results: Iterable[EvalResult], policy: Optional[dict[str, Any]] = None) -> bool:
    return not gate_decision(results, policy)


def write_eval_results(
    trace_path: Path,
    results: list[EvalResult],
    out_path: Optional[Path] = None,
    policy: Optional[dict[str, Any]] = None,
) -> Path:
    out_path = out_path or (trace_path.parent / "eval_results.json")
    payload = {
        "trace_path": str(trace_path),
        "gate_failed": gate_should_fail(results, policy),
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def build_fuzz_cases(
    fuzzer: Callable[[list[int], dict[str, Any]], list[FuzzCase | dict[str, Any]]],
    injection_turns: list[int],
    config: Optional[dict[str, Any]] = None,
) -> list[FuzzCase]:
    config = dict(config or {})
    try:
        raw_cases = fuzzer(injection_turns, config)
    except TypeError:
        raw_cases = fuzzer(injection_turns)
    cases: list[FuzzCase] = []
    for case in raw_cases:
        if isinstance(case, FuzzCase):
            cases.append(case)
        elif isinstance(case, dict):
            payload = dict(case)
            if "scenario_id" not in payload and "case_id" in payload:
                payload["scenario_id"] = payload["case_id"]
            cases.append(FuzzCase(**payload))
        else:
            raise TypeError("Fuzzer must return FuzzCase or dict entries")
    return cases


def suggest_guidance_patch(
    advisor: Callable[[dict[str, Any], list[Any], dict[str, Any]], GuidancePatch | dict[str, Any] | None],
    trace: dict[str, Any],
    eval_results: list[Any],
    config: Optional[dict[str, Any]] = None,
) -> Optional[GuidancePatch]:
    config = dict(config or {})
    try:
        raw = advisor(trace, eval_results, config)
    except TypeError:
        raw = advisor(trace, eval_results)

    if raw is None:
        return None
    if isinstance(raw, GuidancePatch):
        return raw
    if isinstance(raw, dict):
        payload = dict(raw)
        return GuidancePatch(
            patch_id=str(payload.get("patch_id") or ""),
            base_guidance_id=str(payload.get("base_guidance_id") or ""),
            new_fragments=payload.get("new_fragments"),
            diff=payload.get("diff"),
            rationale=str(payload.get("rationale") or ""),
            confidence=payload.get("confidence"),
        )
    raise TypeError("Advisor must return GuidancePatch, dict, or None")


def eval_results_to_dicts(results: Iterable[EvalResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
