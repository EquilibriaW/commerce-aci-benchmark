#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional LLM-based evaluator demo (Anthropic)."""

from __future__ import annotations

import json
import os
from typing import Any

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

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=[{"type": "text", "text": "Return only valid JSON."}],
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
