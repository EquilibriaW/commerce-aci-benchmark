#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM advisor that proposes guidance patches from failing traces."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from pack_api.contracts import GuidancePatch


def _format_eval_results(eval_results: list[Any]) -> list[dict[str, Any]]:
    formatted = []
    for res in eval_results:
        if hasattr(res, "__dict__"):
            payload = res.__dict__
        elif isinstance(res, dict):
            payload = res
        else:
            payload = {"summary": str(res)}
        formatted.append({
            "pack_id": payload.get("pack_id"),
            "evaluator_id": payload.get("evaluator_id"),
            "decision": payload.get("decision"),
            "severity": payload.get("severity"),
            "summary": payload.get("summary"),
        })
    return formatted


def suggest_patch(trace: dict[str, Any], eval_results: list[Any], config: dict[str, Any]) -> GuidancePatch | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    from anthropic import Anthropic

    model = str(config.get("model", "claude-3-5-haiku-20241022"))
    max_fragments = int(config.get("max_fragments", 3))
    base_guidance_id = str(config.get("base_guidance_id", ""))

    meta = trace.get("meta") or {}
    instruction = meta.get("task_instruction") or ""
    task_id = meta.get("task_id") or ""

    eval_summary = _format_eval_results(eval_results)

    prompt = (
        "You are a guidance advisor for a commerce UI agent.\n"
        "Given the task and evaluator outcomes, propose a SMALL guidance patch.\n"
        "Return JSON: {\"fragments\": [\"...\"], \"rationale\": \"...\", \"confidence\": 0-1}.\n"
        f"Max fragments: {max_fragments}. Avoid duplicating existing guidance.\n\n"
        f"Task ID: {task_id}\n"
        f"Task instruction: {instruction}\n\n"
        f"Evaluator results:\n{json.dumps(eval_summary, indent=2)}\n"
    )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=[{"type": "text", "text": "Return only valid JSON."}],
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text = block.text
            break

    try:
        parsed = json.loads(raw_text)
    except Exception:
        return None

    fragments = parsed.get("fragments") or []
    fragments = [str(f).strip() for f in fragments if str(f).strip()]
    if not fragments:
        return None

    rationale = str(parsed.get("rationale") or "")
    confidence = parsed.get("confidence")
    patch_id = f"patch_{uuid.uuid4().hex[:8]}"

    return GuidancePatch(
        patch_id=patch_id,
        base_guidance_id=base_guidance_id,
        new_fragments=fragments,
        diff=None,
        rationale=rationale,
        confidence=confidence if isinstance(confidence, (int, float)) else None,
    )
