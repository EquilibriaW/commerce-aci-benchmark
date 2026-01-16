#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional LLM-based evaluator demo (Anthropic)."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from pack_api.contracts import EvalResult


def goal_adherence(trace: dict[str, Any], config: dict[str, Any]) -> EvalResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return EvalResult(
            pack_id="llm_judge_demo",
            evaluator_id="goal_adherence",
            decision="uncertain",
            severity="warn",
            confidence=None,
            metrics={},
            summary="skipped: ANTHROPIC_API_KEY not set",
            evidence=[],
        )

    from anthropic import Anthropic

    model = str(config.get("model", "claude-3-5-haiku-20241022"))
    min_score = int(config.get("min_score", 3))
    system_prompt = str(config.get("system_prompt", "")).strip()

    meta = trace.get("meta") or {}
    instruction = meta.get("task_instruction") or ""

    last_text = ""
    for step in reversed(trace.get("steps") or []):
        text = ((step.get("assistant") or {}).get("text") or "").strip()
        if text:
            last_text = text
            break

    prompt = (
        "You are a strict evaluator of goal adherence for a commerce agent.\n"
        "Return a JSON object: {\"score\": 1-5, \"summary\": \"...\"}.\n\n"
        f"Task instruction:\n{instruction}\n\n"
        f"Assistant final response:\n{last_text}\n"
    )

    system_text = "Return only valid JSON."
    if system_prompt:
        system_text = f"{system_prompt}\n\n{system_text}"

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=[{"type": "text", "text": system_text}],
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    score = 1
    summary = "unparsed response"
    try:
        parsed = json.loads(raw_text)
        score = int(parsed.get("score", score))
        summary = str(parsed.get("summary", summary))
    except Exception:
        summary = raw_text.strip()[:200] or summary

    decision = "pass" if score >= min_score else "fail"
    return EvalResult(
        pack_id="llm_judge_demo",
        evaluator_id="goal_adherence",
        decision=decision,
        severity="warn",
        confidence=None,
        metrics={"score": score, "min_score": min_score},
        summary=summary,
        evidence=[],
    )


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = stripped[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _summarize_last_steps(trace: dict[str, Any], limit: int = 6) -> str:
    steps = trace.get("steps") or []
    if not steps:
        return "No steps recorded."
    summaries: list[str] = []
    start_idx = max(0, len(steps) - limit)
    for idx in range(start_idx, len(steps)):
        step = steps[idx]
        step_num = idx + 1
        tool_uses = (step.get("assistant") or {}).get("tool_uses") or []
        actions = []
        for tu in tool_uses:
            inp = tu.get("input") or {}
            action = inp.get("action") or "?"
            coord = inp.get("coordinate")
            text = (inp.get("text") or "")[:30]
            if coord:
                actions.append(f"{action}@{coord}")
            elif text:
                actions.append(f"{action}:{text}")
            else:
                actions.append(action)
        assistant_text = ((step.get("assistant") or {}).get("text") or "").strip().replace("\n", " ")
        assistant_text = assistant_text[:160] if assistant_text else ""
        action_str = " | ".join(actions) if actions else "no tool use"
        summaries.append(f"Step {step_num}: {action_str} | {assistant_text}")
    return "\n".join(summaries)


def failure_diagnosis(trace: dict[str, Any], config: dict[str, Any]) -> EvalResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return EvalResult(
            pack_id="llm_judge_demo",
            evaluator_id="failure_diagnosis",
            decision="uncertain",
            severity="warn",
            confidence=None,
            metrics={},
            summary="skipped: ANTHROPIC_API_KEY not set",
            evidence=[],
        )

    from anthropic import Anthropic

    model = str(config.get("model", "claude-3-5-haiku-20241022"))
    rubric = str(config.get("rubric", "")).strip() or (
        "Diagnose likely failure modes, the most actionable fix, and whether this run should be marked as fail or uncertain."
    )
    system_prompt = str(config.get("system_prompt", "")).strip()

    meta = trace.get("meta") or {}
    instruction = meta.get("task_instruction") or ""
    final_state = meta.get("final_state") or {}
    stage = final_state.get("stage") or "unknown"
    progress = final_state.get("progress") or {}
    events = final_state.get("events") or {}

    prompt = (
        "You are a diagnostic evaluator for a commerce GUI agent.\n"
        "Return only valid JSON with keys:\n"
        "{\n"
        "  \"verdict\": \"fail\" | \"uncertain\" | \"pass\",\n"
        "  \"confidence\": 0-1,\n"
        "  \"summary\": \"one-paragraph diagnosis\",\n"
        "  \"failure_reason\": \"short tag\",\n"
        "  \"suggested_intervention\": \"concrete, actionable fix\",\n"
        "  \"evidence\": [{\"step_index\": int, \"note\": \"...\"}]\n"
        "}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Task instruction:\n{instruction}\n\n"
        f"Final state (stage={stage}):\n{json.dumps(final_state, ensure_ascii=True)}\n\n"
        f"Progress flags:\n{json.dumps(progress, ensure_ascii=True)}\n\n"
        f"Event counts:\n{json.dumps(events.get('counts_by_type') or {}, ensure_ascii=True)}\n\n"
        "Recent steps:\n"
        f"{_summarize_last_steps(trace)}\n"
    )

    client = Anthropic(api_key=api_key)
    system_text = "Return only valid JSON."
    if system_prompt:
        system_text = f"{system_prompt}\n\n{system_text}"

    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=[{"type": "text", "text": system_text}],
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    parsed = _extract_json(raw_text) or {}
    verdict = str(parsed.get("verdict", "uncertain")).lower()
    if verdict not in {"pass", "fail", "uncertain"}:
        verdict = "uncertain"
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None

    summary = str(parsed.get("summary", "")).strip() or "diagnosis unavailable"
    failure_reason = str(parsed.get("failure_reason", "")).strip()
    suggested_intervention = str(parsed.get("suggested_intervention", "")).strip()
    evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else []
    evidence_items = []
    for item in evidence:
        if isinstance(item, dict):
            evidence_items.append({
                "step_index": item.get("step_index"),
                "note": item.get("note"),
            })

    metrics = {
        "failure_reason": failure_reason,
        "suggested_intervention": suggested_intervention,
        "rubric": rubric,
    }

    return EvalResult(
        pack_id="llm_judge_demo",
        evaluator_id="failure_diagnosis",
        decision=verdict,
        severity="warn",
        confidence=confidence,
        metrics=metrics,
        summary=summary,
        evidence=evidence_items,
    )
