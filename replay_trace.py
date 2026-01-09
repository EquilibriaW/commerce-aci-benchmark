#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic trace replay + counterfactual (shadow) analysis.

This script turns a single benchmark run into a reusable artifact:

1) **Re-execution replay** (no LLM required)
   Replays the recorded UI actions in Playwright to reproduce a run.

2) **Shadow counterfactual** (LLM required)
   Replays the recorded *observations* (screenshots) while letting a different
   model/prompt propose actions, and reports divergence/action-agreement.

Why this matters:
- This is the minimal "VCR for agents" primitive: reproduce failures, then
  run counterfactuals against a frozen world.

Usage
-----

Re-execute a trace (reproduce a run):

    python replay_trace.py --trace path/to/trace.json --mode reexecute

Shadow counterfactual (compare a new prompt/model against a frozen observation sequence):

    export ANTHROPIC_API_KEY=sk-ant-...
    python replay_trace.py --trace path/to/trace.json --mode shadow \
        --model claude-sonnet-4-5-20250929 \
        --system-prompt-file my_system_prompt.txt

Outputs
-------
- For shadow mode: writes a JSON report next to the trace: counterfactual_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import async_playwright

from benchmark_computeruse import (
    BENCHMARK_SECRET,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    TASKS,
    ComputerUseAgent,
)


def _load_trace(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "trace.v1":
        raise ValueError(f"Unsupported trace schema_version: {payload.get('schema_version')}")
    if "meta" not in payload or "steps" not in payload:
        raise ValueError("Invalid trace: missing meta/steps")
    return payload


def _b64_png(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.standard_b64encode(data).decode("utf-8")


def _find_task(task_id: str) -> dict[str, Any]:
    for t in TASKS:
        if t.get("id") == task_id:
            return t
    raise KeyError(f"Task id not found in TASKS: {task_id}")


@dataclass
class ActionSig:
    action: str
    text: Optional[str] = None
    coordinate: Optional[tuple[int, int]] = None


def _action_signature(tool_input: dict[str, Any]) -> ActionSig:
    action = str(tool_input.get("action", "unknown"))
    text = tool_input.get("text")
    coord = tool_input.get("coordinate")
    if isinstance(coord, list) and len(coord) == 2:
        try:
            coordinate = (int(coord[0]), int(coord[1]))
        except Exception:
            coordinate = None
    else:
        coordinate = None
    return ActionSig(action=action, text=text, coordinate=coordinate)


def _first_action_from_step(step: dict[str, Any]) -> Optional[ActionSig]:
    tool_uses = ((step.get("assistant") or {}).get("tool_uses") or [])
    if not tool_uses:
        return None
    first = tool_uses[0]
    return _action_signature(first.get("input") or {})


async def reexecute_trace(trace: dict[str, Any], output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Re-run recorded UI actions in Playwright (no LLM calls)."""
    meta = trace["meta"]
    task_id = meta.get("task_id")
    if not task_id:
        raise ValueError("Trace meta missing task_id")

    task = _find_task(task_id)

    target_url = meta.get("target_url")
    api_url = meta.get("api_url")
    discoverability = meta.get("discoverability", "navbar")
    capability = meta.get("capability", "advantage")

    if not target_url or not api_url:
        raise ValueError("Trace meta missing target_url/api_url")

    # Optional output directory for replay artifacts
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"]
        )
        context = await browser.new_context(
            viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
            user_agent="CommerceACIBenchmark/Replay/1.0",
        )
        page = await context.new_page()
        agent = ComputerUseAgent(page, api_key=None, debug_dir=output_dir)

        # Reset to reproduce the same initial state
        await agent.reset_session(api_url, discoverability=discoverability, capability=capability)
        await page.goto(target_url)
        await page.wait_for_load_state("networkidle")

        total_actions = 0
        for step in trace["steps"]:
            tool_uses = ((step.get("assistant") or {}).get("tool_uses") or [])
            for idx, tu in enumerate(tool_uses, start=1):
                tool_input = tu.get("input") or {}
                # Execute using the same low-level action executor as the benchmark
                await agent._execute_computer_action(tool_input, action_index=idx)
                total_actions += 1

        final_state = await agent.get_ground_truth(api_url)
        success = bool(final_state and task["verifier"](final_state))

        await browser.close()

    return {
        "mode": "reexecute",
        "task_id": task_id,
        "success": success,
        "total_actions": total_actions,
        "final_state": final_state,
    }


async def shadow_counterfactual(
    trace: dict[str, Any],
    model: str,
    system_prompt: str,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Run a counterfactual policy on a frozen observation sequence.

    We replay the recorded screenshots and provide them as tool_results
    regardless of the policy's proposed actions. This is an "off-policy"
    evaluation that is useful for:

    - action agreement / divergence detection
    - prompt/model comparisons on the same visited states
    """
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for --mode shadow")

    client = Anthropic(api_key=api_key)
    steps = trace["steps"]
    meta = trace["meta"]
    task_instruction = meta.get("task_instruction") or ""

    # Rebuild a conversation with computer tool.
    messages: list[dict[str, Any]] = []

    # Initial message uses step 1 observation.
    if not steps:
        raise ValueError("Trace has no steps")
    first_obs_path = (steps[0].get("observation") or {}).get("screenshot_path")
    if not first_obs_path:
        raise ValueError("Trace step 1 missing observation.screenshot_path")

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"TASK: {task_instruction}\n\nHere is the current state of the webpage:"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _b64_png(first_obs_path)}
            },
        ],
    })

    predicted_actions: list[dict[str, Any]] = []
    original_actions: list[dict[str, Any]] = []

    # Iterate through recorded steps and replay observations.
    for i, step in enumerate(steps, start=1):
        if max_steps and i > max_steps:
            break

        # For steps > 1, append the recorded observation (pre-screenshot)
        if i > 1:
            obs_path = (step.get("observation") or {}).get("screenshot_path")
            if obs_path:
                prefix = ""
                extra = step.get("user_extra")
                if extra:
                    prefix = f"USER UPDATE: {extra}\n\n"
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prefix + "Here is the current state of the page after your last action. Continue with the task or say TASK_COMPLETE if done."},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": _b64_png(obs_path)}
                        },
                    ],
                })

        # Call model
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=2048,
            system=system_prompt,
            extra_headers={"anthropic-beta": "computer-use-2025-01-24"},
            tools=[
                {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": DISPLAY_WIDTH,
                    "display_height_px": DISPLAY_HEIGHT,
                    "display_number": 1,
                }
            ],
            messages=messages,
        )

        assistant_content: list[dict[str, Any]] = []
        tool_uses: list[dict[str, Any]] = []
        assistant_text: list[str] = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                assistant_text.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        # Record predicted vs original first action signature at this step
        predicted_first = _action_signature(tool_uses[0]["input"]) if tool_uses else None
        original_first = _first_action_from_step(step)
        predicted_actions.append({
            "step": i,
            "assistant_text": "\n".join(assistant_text) if assistant_text else None,
            "first_action": predicted_first.__dict__ if predicted_first else None,
        })
        original_actions.append({
            "step": i,
            "first_action": original_first.__dict__ if original_first else None,
        })

        # Provide tool_result(s) using recorded tool_results screenshots.
        recorded_results = step.get("tool_results") or []
        if tool_uses:
            tool_results_blocks: list[dict[str, Any]] = []
            for j, tu in enumerate(tool_uses):
                # Pick a recorded result to return.
                rec = recorded_results[j] if j < len(recorded_results) else (recorded_results[-1] if recorded_results else None)
                rec_text = (rec or {}).get("result_text") or "(replayed)"
                rec_img_path = (rec or {}).get("screenshot_path")
                content: list[dict[str, Any]] = [{"type": "text", "text": rec_text}]
                if rec_img_path and Path(rec_img_path).exists():
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": _b64_png(rec_img_path)}
                    })
                tool_results_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": content,
                })
            messages.append({"role": "user", "content": tool_results_blocks})

    # Compute agreement + first divergence
    agreement = 0
    comparable = 0
    first_divergence_step: Optional[int] = None
    for p, o in zip(predicted_actions, original_actions):
        p_act = p.get("first_action")
        o_act = o.get("first_action")
        if p_act is None or o_act is None:
            continue
        comparable += 1
        if p_act.get("action") == o_act.get("action"):
            # For TYPE, also compare prefix of text; for CLICK compare coordinate exact match.
            if p_act.get("action") == "type":
                p_text = (p_act.get("text") or "")
                o_text = (o_act.get("text") or "")
                if p_text[:20] == o_text[:20]:
                    agreement += 1
                elif first_divergence_step is None:
                    first_divergence_step = p["step"]
            elif p_act.get("action") in {"left_click", "right_click", "double_click", "triple_click"}:
                if p_act.get("coordinate") == o_act.get("coordinate"):
                    agreement += 1
                elif first_divergence_step is None:
                    first_divergence_step = p["step"]
            else:
                agreement += 1
        elif first_divergence_step is None:
            first_divergence_step = p["step"]

    agreement_rate = (agreement / comparable) if comparable else None

    return {
        "mode": "shadow",
        "model": model,
        "task_id": meta.get("task_id"),
        "task_instruction": task_instruction,
        "steps_evaluated": len(predicted_actions),
        "comparable_steps": comparable,
        "first_action_agreement": agreement_rate,
        "first_divergence_step": first_divergence_step,
        "predicted_actions": predicted_actions,
        "original_actions": original_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a benchmark trace.")
    parser.add_argument("--trace", type=str, required=True, help="Path to trace.json")
    parser.add_argument("--mode", choices=["reexecute", "shadow"], required=True)

    # Shadow mode options
    parser.add_argument("--model", type=str, default=None, help="Anthropic model name (shadow mode)")
    parser.add_argument("--system-prompt-file", type=str, default=None, help="Optional path to system prompt text")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit number of steps evaluated")

    # Reexecute mode options
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store replay artifacts")

    args = parser.parse_args()
    trace_path = Path(args.trace)
    trace = _load_trace(trace_path)

    if args.mode == "reexecute":
        out_dir = Path(args.output_dir) if args.output_dir else None
        res = asyncio.run(reexecute_trace(trace, output_dir=out_dir))
        print(json.dumps(res, indent=2))
        return

    # shadow mode
    model = args.model or trace["meta"].get("model") or "claude-sonnet-4-5-20250929"
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    else:
        system_prompt = trace["meta"].get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    res = asyncio.run(shadow_counterfactual(trace, model=model, system_prompt=system_prompt, max_steps=args.max_steps))
    # Save next to trace
    out_path = trace_path.parent / f"counterfactual_{model.replace('/', '_')}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(json.dumps({k: res[k] for k in ["task_id", "model", "steps_evaluated", "first_action_agreement", "first_divergence_step"]}, indent=2))


if __name__ == "__main__":
    main()
