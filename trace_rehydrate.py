#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rehydrate a trace into a ComputerUseAgent-compatible message history."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional


def _read_image_b64(path_str: Optional[str], base_dir: Optional[Path]) -> Optional[str]:
    if not path_str:
        return None
    path = Path(path_str)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        if base_dir:
            candidates.insert(0, base_dir / path)
    for candidate in candidates:
        try:
            if candidate.exists():
                return base64.standard_b64encode(candidate.read_bytes()).decode("utf-8")
        except Exception:
            continue
    return None


def _image_block(b64: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64,
        },
    }


def build_conversation_from_steps(
    steps: list[dict[str, Any]],
    instruction: str,
    trace_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Convert trace steps into a message history compatible with ComputerUseAgent."""
    messages: list[dict[str, Any]] = []
    last_was_tool_result = False

    for idx, step in enumerate(steps):
        user_extra = step.get("user_extra")
        observation = step.get("observation") or {}

        if last_was_tool_result and user_extra:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_extra}],
            })

        if not last_was_tool_result:
            if idx == 0:
                extra = f"\n\nUSER UPDATE: {user_extra}" if user_extra else ""
                text = f"TASK: {instruction}{extra}\n\nHere is the current state of the webpage:"
            else:
                prefix = f"USER UPDATE: {user_extra}\n\n" if user_extra else ""
                text = (
                    prefix
                    + "Here is the current state of the page after your last action. "
                    "Continue with the task or say TASK_COMPLETE if done."
                )

            content = [{"type": "text", "text": text}]
            b64 = _read_image_b64(observation.get("screenshot_path"), trace_dir)
            if b64:
                content.append(_image_block(b64))
            messages.append({"role": "user", "content": content})

        assistant = step.get("assistant") or {}
        assistant_content = []
        assistant_text = assistant.get("text")
        if assistant_text:
            assistant_content.append({"type": "text", "text": assistant_text})

        for tu in (assistant.get("tool_uses") or []):
            assistant_content.append({
                "type": "tool_use",
                "id": tu.get("tool_use_id") or tu.get("id") or "",
                "name": tu.get("name") or "computer",
                "input": tu.get("input") or {},
            })

        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        tool_results = step.get("tool_results") or []
        if tool_results:
            result_blocks = []
            for tr in tool_results:
                result_text = tr.get("result_text") or ""
                tr_content = [{"type": "text", "text": result_text}]
                b64 = _read_image_b64(tr.get("screenshot_path"), trace_dir)
                if b64:
                    tr_content.append(_image_block(b64))
                result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tr.get("tool_use_id") or "",
                    "content": tr_content,
                })
            messages.append({"role": "user", "content": result_blocks})
            last_was_tool_result = True
        else:
            last_was_tool_result = False

    return messages


def build_conversation_from_trace(
    trace: dict[str, Any],
    trace_dir: Optional[Path] = None,
    up_to_step: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Convert trace dict into a message history up to a given step."""
    meta = trace.get("meta") or {}
    instruction = meta.get("task_instruction") or "Continue the task"
    steps = trace.get("steps") or []
    if up_to_step is None:
        up_to_step = len(steps)
    return build_conversation_from_steps(steps[:up_to_step], instruction, trace_dir=trace_dir)
