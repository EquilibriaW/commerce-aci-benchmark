#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentOps Control Panel - Trace Viewer & Branching.

Features
--------
1) Run Launcher
   - Run benchmark tasks or custom goals
   - Configure environment, model, and system prompt

2) Trace Viewer
   - Step-by-step trace inspection
   - Shows branch lineage for branched traces

3) Trace Trees
   - Create and explore trace branches
   - Test counterfactuals with different prompts, models, or actions

Usage
-----
    pip install -r requirements.txt
    streamlit run control_panel.py

Notes
-----
- Requires ANTHROPIC_API_KEY for running agents.
- Requires Playwright browsers: playwright install chromium
"""

from __future__ import annotations

import asyncio
import json
import difflib
import hashlib
import os
import subprocess
import textwrap
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from playwright.async_api import async_playwright

try:
    import streamlit as st
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "\nThis control panel requires Streamlit.\n\n"
        "Install it with:\n"
        "  pip install streamlit pandas pillow\n\n"
        f"Import error: {e}\n"
    )

from replay_trace import _load_trace, _first_action_from_step
from benchmark_computeruse import (
    ANTHROPIC_API_KEY,
    ASK_USER_TAG,
    DEBUG_DIR as DEFAULT_DEBUG_DIR,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    MAX_ITERATIONS,
    TASKS as PREDEFINED_TASKS,
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    URL_BASELINE,
    URL_TREATMENT,
    URL_TREATMENT_DOCS,
    ComputerUseAgent,
)

# Import branching modules
from trace_tree import TraceTree, Intervention, InterventionType
from branch_executor import BranchExecutionConfig, run_branch_sync
from tree_visualization import render_trace_tree
from pack_api.loader import PackRegistry
from pack_api.runtime import (
    assemble_system_prompt,
    gate_decision,
    load_guidance_fragments,
    run_evaluators_on_trace,
    suggest_guidance_patch,
)


@dataclass
class TraceItem:
    label: str
    path: Path
    meta: dict[str, Any]


def _safe_read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_traces(root: Path) -> list[TraceItem]:
    traces: list[TraceItem] = []
    if not root.exists():
        return traces

    for p in sorted(root.rglob("trace.json")):
        payload = _safe_read_json(p)
        if not payload:
            continue
        schema = payload.get("schema_version", "")
        if schema != "trace.v2":
            continue
        meta = payload.get("meta") or {}
        cond = meta.get("condition_name") or meta.get("condition") or "(unknown condition)"
        task = meta.get("task_id") or meta.get("scenario_id") or "(unknown task)"
        ts = meta.get("run_timestamp") or p.parent.name
        label = f"{cond} | {task} | {ts}"
        traces.append(TraceItem(label=label, path=p, meta=meta))
    return traces


def _img_exists(p: Optional[str]) -> bool:
    return bool(p) and Path(p).exists()


def _find_tree_for_trace(trace_id: str, trees_dir: Path) -> Optional[TraceTree]:
    if not trace_id or not trees_dir.exists():
        return None
    for info in TraceTree.list_trees(trees_dir):
        try:
            tree = TraceTree(info["tree_id"], trees_dir)
        except Exception:
            continue
        if trace_id in tree._index.traces:
            return tree
    return None


def _action_matches(a, b) -> bool:
    if a is None or b is None:
        return True
    if a.action != b.action:
        return False
    if a.action == "type":
        return (a.text or "")[:20] == (b.text or "")[:20]
    if a.action in {"left_click", "right_click", "double_click", "triple_click"}:
        return a.coordinate == b.coordinate
    return True


def _first_divergence_step(left_steps: list[dict], right_steps: list[dict]) -> Optional[int]:
    for idx, (left, right) in enumerate(zip(left_steps, right_steps), start=1):
        left_act = _first_action_from_step(left)
        right_act = _first_action_from_step(right)
        if left_act is None or right_act is None:
            continue
        if not _action_matches(left_act, right_act):
            return idx
    return None


def _extract_tool_actions(steps: list[dict]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        tool_uses = (step.get("assistant") or {}).get("tool_uses") or []
        for tu in tool_uses:
            inp = tu.get("input") or {}
            action = inp.get("action") or "?"
            actions.append({
                "step": idx,
                "action": action,
                "coordinate": inp.get("coordinate"),
                "text": (inp.get("text") or "")[:40],
            })
    return actions


def _action_key(action: dict[str, Any]) -> tuple:
    coord = action.get("coordinate")
    coord_key = tuple(coord) if isinstance(coord, list) else None
    return (action.get("action"), coord_key, action.get("text") or "")


def _detect_action_loop(actions: list[dict[str, Any]], min_run: int = 3) -> Optional[dict[str, Any]]:
    if not actions:
        return None
    best = None
    current_key = None
    current_start = 0
    current_len = 0

    for idx, action in enumerate(actions):
        key = _action_key(action)
        if key == current_key:
            current_len += 1
        else:
            if current_len >= min_run:
                best = (current_start, current_len, current_key)
            current_key = key
            current_start = idx
            current_len = 1

    if current_len >= min_run:
        best = (current_start, current_len, current_key)

    if not best:
        return None

    start_idx, length, key = best
    start_step = actions[start_idx]["step"]
    end_step = actions[start_idx + length - 1]["step"]
    return {
        "action": key[0],
        "coordinate": key[1],
        "text": key[2],
        "start_step": start_step,
        "end_step": end_step,
        "length": length,
    }


def _action_counts(actions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        name = action.get("action") or "?"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _task_complete_mentioned(steps: list[dict]) -> bool:
    for step in reversed(steps):
        text = (step.get("assistant") or {}).get("text") or ""
        if "TASK_COMPLETE" in text:
            return True
    return False


def _persist_judge_prompt(
    trace_path: Path,
    trace: dict[str, Any],
    prompt: str,
    label: str,
) -> None:
    if not prompt:
        return
    meta = trace.get("meta") or {}
    prompts = dict(meta.get("judge_system_prompts") or {})
    hashes = dict(meta.get("judge_system_prompt_hashes") or {})
    prompts[label] = prompt
    hashes[label] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    meta["judge_system_prompts"] = prompts
    meta["judge_system_prompt_hashes"] = hashes
    trace["meta"] = meta
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def _derive_failure_reason(final_state: dict[str, Any], success_flag: Optional[bool]) -> str:
    if success_flag:
        return "success"
    stage = (final_state or {}).get("stage")
    if stage == "browse":
        return "no_items_added"
    if stage == "cart":
        return "checkout_not_started"
    if stage == "checkout":
        return "checkout_incomplete"
    if stage == "done":
        return "verifier_failed"
    if stage == "error":
        return "state_error"
    return "unknown"


class InteractiveRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.agent: Optional[ComputerUseAgent] = None
        self.started = False
        self.done = False
        self.success = False
        self.steps = 0
        self.last_step: Optional[dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.trace_path: Optional[Path] = None
        self.trace_meta: dict[str, Any] = {}
        self.max_iterations = MAX_ITERATIONS
        self.instruction = ""
        self.verifier = None
        self.api_url = ""
        self.target_url = ""
        self.condition_name = ""
        self.discoverability = ""
        self.capability = ""

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _submit(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def start(
        self,
        *,
        instruction: str,
        verifier,
        condition_name: str,
        target_url: str,
        api_url: str,
        discoverability: str,
        capability: str,
        system_prompt: str,
        model: str,
        debug_dir: Path,
        trace_meta: dict[str, Any],
    ) -> dict[str, Any]:
        self.instruction = instruction
        self.verifier = verifier
        self.condition_name = condition_name
        self.target_url = target_url
        self.api_url = api_url
        self.discoverability = discoverability
        self.capability = capability
        self.trace_path = debug_dir / "trace.json"
        self.trace_meta = dict(trace_meta)
        return self._submit(self._start_async(system_prompt, model, debug_dir))

    async def _start_async(self, system_prompt: str, model: str, debug_dir: Path) -> dict[str, Any]:
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"],
        )
        self.context = await self.browser.new_context(
            viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
            user_agent="CommerceACIBenchmark/Interactive/1.0",
        )
        self.page = await self.context.new_page()
        self.agent = ComputerUseAgent(
            self.page,
            ANTHROPIC_API_KEY,
            debug_dir=debug_dir,
            system_prompt=system_prompt,
            model=model,
        )
        await self.agent.reset_session(self.api_url, discoverability=self.discoverability, capability=self.capability)
        await self.page.goto(self.target_url)
        await self.page.wait_for_load_state("networkidle")
        self.started = True
        self.done = False
        self.success = False
        self.steps = 0
        self.last_step = None
        self._write_trace()
        return {"status": "started"}

    def step(self, user_message: str | None = None) -> dict[str, Any]:
        return self._submit(self._step_async(user_message))

    async def _step_async(self, user_message: str | None) -> dict[str, Any]:
        if not self.agent or self.done:
            return {"done": self.done, "success": self.success}

        if self.verifier:
            state = await self.agent.get_ground_truth(self.api_url)
            if state and self.verifier(state):
                self.success = True
                self.done = True
                self._write_trace()
                return {"done": True, "success": True}

        action, is_done = await self.agent.run_step(self.instruction, extra_user_text=user_message)
        self.steps += 1
        self.last_step = self.agent.trace["steps"][-1] if self.agent.trace.get("steps") else None

        if is_done:
            if self.verifier:
                state = await self.agent.get_ground_truth(self.api_url)
                self.success = bool(state and self.verifier(state))
            else:
                self.success = True
            self.done = True

        if self.steps >= self.max_iterations:
            self.done = True

        self._write_trace()
        return {
            "done": self.done,
            "success": self.success,
            "steps": self.steps,
            "last_step": self.last_step,
            "action": action,
        }

    def stop(self) -> None:
        self._submit(self._stop_async())

    async def _stop_async(self) -> None:
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.started = False

    def _write_trace(self) -> None:
        if not self.agent or not self.trace_path:
            return
        meta = dict(self.trace_meta)
        meta["success"] = self.success
        self.trace_path.write_text(
            json.dumps(self.agent.export_trace(meta), indent=2),
            encoding="utf-8",
        )


def main() -> None:
    st.set_page_config(page_title="AgentOps Control Panel", layout="wide")
    st.title("AgentOps Control Panel")

    if "interactive_runner" not in st.session_state:
        st.session_state["interactive_runner"] = None
    if "interactive_trace_path" not in st.session_state:
        st.session_state["interactive_trace_path"] = ""
    if "interactive_user_message" not in st.session_state:
        st.session_state["interactive_user_message"] = ""
    if "trace_path_override" not in st.session_state:
        st.session_state["trace_path_override"] = ""
    if "reliability_report" not in st.session_state:
        st.session_state["reliability_report"] = None
    if "reliability_out_path" not in st.session_state:
        st.session_state["reliability_out_path"] = ""
    if "eval_judge_prompt" not in st.session_state:
        st.session_state["eval_judge_prompt"] = ""
    if "diag_system_prompt" not in st.session_state:
        st.session_state["diag_system_prompt"] = st.session_state.get("eval_judge_prompt", "")

    # Sidebar config
    st.sidebar.header("Trace Source")
    debug_root = st.sidebar.text_input("Trace directory", value=str(DEFAULT_DEBUG_DIR))
    debug_root_p = Path(debug_root)

    traces = find_traces(debug_root_p)

    # Always show tabs - Run Launcher doesn't require traces
    tabs = st.tabs(["Run Launcher", "Packs & Guidance", "Trace Viewer", "Trace Trees", "Reliability Dashboard"])

    # Trace selection (only if traces exist)
    trace_item = None
    trace = None
    meta = {}
    steps = []

    if traces:
        trace_labels = [t.label for t in traces]
        selected_index = 0
        override_path = st.session_state.get("trace_path_override")
        if override_path:
            for idx, item in enumerate(traces):
                if str(item.path) == str(override_path):
                    selected_index = idx
                    st.session_state["trace_path_override"] = ""
                    break
        choice = st.sidebar.selectbox(
            "Select a trace",
            options=list(range(len(traces))),
            format_func=lambda i: trace_labels[i],
            index=selected_index
        )
        trace_item = traces[int(choice)]
        trace = _load_trace(trace_item.path)
        meta = trace.get("meta") or {}
        steps = trace.get("steps") or []

        st.sidebar.subheader("Trace Meta")
        st.sidebar.json(meta)
    else:
        st.sidebar.warning("No traces found. Run a benchmark first to generate traces.")

    # === Run Launcher (Tab 0) ===
    with tabs[0]:
        st.subheader("Run Launcher")
        st.caption("Run the agent with predefined benchmark tasks or a custom goal.")

        # --- Task Mode Selection ---
        st.markdown("### What to Run")
        task_mode = st.radio(
            "Task mode",
            options=["Custom Goal", "Predefined Tasks", "Fuzz Testing"],
            index=0,
            horizontal=True,
            help="Custom Goal: type any instruction. Predefined Tasks: select from benchmark tasks. Fuzz: adversarial testing."
        )

        custom_instruction = ""
        selected_task_ids: list[str] = []

        if task_mode == "Custom Goal":
            custom_instruction = st.text_area(
                "Enter your goal/instruction",
                placeholder="e.g., Buy me a red hoodie in size medium",
                height=100,
                help="The agent will attempt to complete this goal. No automated verification - runs until agent says done or hits max iterations."
            )
        elif task_mode == "Predefined Tasks":
            # Build task options for multiselect
            task_options = {t["id"]: f"{t['id']}: {t['instruction'][:50]}..." for t in PREDEFINED_TASKS}
            selected_task_ids = st.multiselect(
                "Select tasks to run",
                options=list(task_options.keys()),
                format_func=lambda x: task_options[x],
                default=[],
                help="Select one or more predefined benchmark tasks. These have automated verifiers."
            )
            if not selected_task_ids:
                st.info("Select at least one task, or switch to Custom Goal mode.")
        # Fuzz mode doesn't need task selection

        # --- Environment Configuration ---
        st.markdown("### Environment")
        col1, col2 = st.columns(2)
        with col1:
            app = st.selectbox(
                "App",
                options=["treatment", "treatment-docs", "baseline"],
                index=0,
                help="baseline: standard e-commerce UI. treatment: adds agent terminal UI. treatment-docs: adds documentation-style agent UI."
            )
            start = st.selectbox(
                "Start point",
                options=["root", "agent"],
                index=0,
                help="root: agent starts at homepage (must discover agent UI). agent: agent starts directly on /agent page (forced adoption)."
            )
        with col2:
            discoverability = st.selectbox(
                "Discoverability",
                options=["navbar", "hidden"],
                index=0,
                help="navbar: agent UI link visible in navigation. hidden: no visible link (agent must find it another way)."
            )
            capability = st.selectbox(
                "Capability",
                options=["advantage", "parity"],
                index=0,
                help="advantage: agent UI has interactive controls (add to cart, checkout). parity: agent UI is read-only (must use human UI to act)."
            )

        col3, col4 = st.columns(2)
        with col3:
            model = st.selectbox(
                "Model",
                options=["sonnet", "haiku"],
                index=0,
                help="sonnet: Claude Sonnet 4.5 ($3/$15 per MTok) - more capable. haiku: Claude Haiku 3.5 ($0.80/$4 per MTok) - faster & cheaper."
            )
        with col4:
            runs_per_task = st.number_input(
                "Runs per task",
                min_value=1,
                max_value=20,
                value=1,
                help="Number of times to repeat each task. Useful for measuring variance across runs."
            )

        # --- System Prompt Override ---
        with st.expander("System Prompt Override (optional)"):
            prompt_override = st.text_area(
                "Custom system prompt",
                value="",
                height=160,
                help="Leave empty to use the default system prompt. This overrides the agent's instructions."
            )

        # --- Packs & Guidance ---
        st.markdown("### Packs & Guidance")
        registry = None
        packs = []
        try:
            registry = PackRegistry.discover()
            packs = registry.list_packs()
        except Exception as e:
            st.warning(f"Failed to load packs: {e}")

        guidance_pack_ids = [p.id for p in packs if p.guidance]
        eval_pack_ids = [p.id for p in packs if p.evaluators]
        selected_guidance_packs = st.multiselect("Guidance Packs", options=guidance_pack_ids, default=[])
        selected_eval_packs = st.multiselect("Eval Packs", options=eval_pack_ids, default=[])

        base_prompt_preview = prompt_override.strip() or DEFAULT_SYSTEM_PROMPT
        guidance_fragments: list[str] = []
        if registry and selected_guidance_packs:
            try:
                guidance_fragments = load_guidance_fragments(registry, selected_guidance=selected_guidance_packs)
            except Exception as e:
                st.warning(f"Failed to load guidance fragments: {e}")
        assembled_prompt = assemble_system_prompt(base_prompt_preview, guidance_fragments)

        st.markdown("### Run Mode")
        run_mode = st.radio(
            "Run mode",
            options=["Batch (CLI)", "Interactive (human-in-the-loop)"],
            index=0,
            horizontal=True,
            help="Batch runs launch a full benchmark process. Interactive runs step the agent inside this UI.",
        )
        interactive_mode = run_mode.startswith("Interactive")
        interactive_instructions = ""
        if interactive_mode:
            interactive_instructions = (
                "\n\n### User Clarifications\n"
                "If you need clarification or a user decision, respond with a single line starting with "
                f"'{ASK_USER_TAG}' followed by your question. After asking, wait for user input."
            )

        assembled_prompt_effective = assembled_prompt + interactive_instructions
        with st.expander("Assembled system prompt preview"):
            st.text_area(
                "prompt_preview",
                assembled_prompt_effective,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

        # --- Build Command ---
        cmd: list[str] = ["python"]
        prompt_file = None

        if task_mode == "Fuzz Testing":
            cmd += ["flow_fuzz.py",
                    "--app", app,
                    "--start", start,
                    "--discoverability", discoverability,
                    "--capability", capability]
            # Note: flow_fuzz.py may need --model support added separately
        else:
            cmd += ["benchmark_computeruse.py",
                    "--app", app,
                    "--start", start,
                    "--discoverability", discoverability,
                    "--capability", capability,
                    "--model", model,
                    "--runs-per-task", str(int(runs_per_task))]

            if task_mode == "Custom Goal" and custom_instruction.strip():
                # Use --instruction for ad-hoc tasks
                cmd += ["--instruction", custom_instruction.strip()]
            elif task_mode == "Predefined Tasks" and selected_task_ids:
                cmd += ["--tasks", ",".join(selected_task_ids)]

        if prompt_override.strip() and not interactive_mode:
            tmp_dir = Path(".tmp_prompts")
            tmp_dir.mkdir(exist_ok=True)
            prompt_file = tmp_dir / f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            prompt_file.write_text(prompt_override, encoding="utf-8")
            cmd += ["--system-prompt-file", str(prompt_file)]

        if selected_guidance_packs:
            cmd += ["--guidance-packs", ",".join(selected_guidance_packs)]
        if selected_eval_packs:
            cmd += ["--eval-packs", ",".join(selected_eval_packs)]

        # --- Show Command & Run Button ---
        st.markdown("### Command")
        if interactive_mode:
            st.caption("Batch command preview (interactive runs execute inside this UI).")
        st.code(" ".join(cmd))

        # Validation
        can_run = True
        if task_mode == "Custom Goal" and not custom_instruction.strip():
            can_run = False
            st.warning("Enter a goal/instruction to run.")
        elif task_mode == "Predefined Tasks" and not selected_task_ids:
            can_run = False

        if not interactive_mode and can_run and st.button("Run now", type="primary"):
            with st.spinner("Running… (this may take a while)"):
                p = subprocess.run(cmd, capture_output=True, text=True)
            st.markdown("**STDOUT**")
            st.text_area("stdout", value=p.stdout, height=240)
            st.markdown("**STDERR**")
            st.text_area("stderr", value=p.stderr, height=240)
            if p.returncode != 0:
                st.error(f"Command failed with exit code {p.returncode}")
            else:
                st.success("Run completed. Check Trace Viewer tab to see results.")

        if prompt_file:
            st.caption(f"Wrote prompt override to: {prompt_file}")

        if interactive_mode:
            st.markdown("### Interactive Session")
            st.caption("Step the agent, inject user follow-ups, and inspect the live trace.")

            if task_mode == "Fuzz Testing":
                st.warning("Interactive mode does not support fuzz testing.")
            elif not ANTHROPIC_API_KEY:
                st.error("ANTHROPIC_API_KEY is required for interactive runs.")
            else:
                instruction = ""
                verifier = None
                task_id = ""
                interactive_ready = True

                if task_mode == "Custom Goal":
                    instruction = custom_instruction.strip()
                    if not instruction:
                        interactive_ready = False
                        st.warning("Enter a goal/instruction to start an interactive run.")
                    task_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                elif task_mode == "Predefined Tasks":
                    if len(selected_task_ids) != 1:
                        interactive_ready = False
                        st.warning("Select exactly one predefined task for interactive runs.")
                    else:
                        task_id = selected_task_ids[0]
                        task = next(t for t in PREDEFINED_TASKS if t["id"] == task_id)
                        instruction = task["instruction"]
                        verifier = task.get("verifier")

                base_url = {
                    "baseline": URL_BASELINE,
                    "treatment": URL_TREATMENT,
                    "treatment-docs": URL_TREATMENT_DOCS,
                }[app]
                target_url = base_url + ("/agent" if start == "agent" else "")
                api_url = base_url
                condition_name = f"Interactive/{app}/{start}/{discoverability}/{capability}"

                runner = st.session_state.get("interactive_runner")
                if runner and runner.started:
                    cols = st.columns(4)
                    cols[0].metric("Steps", runner.steps)
                    cols[1].metric("Done", runner.done)
                    cols[2].metric("Success", runner.success)
                    cols[3].metric("Max", runner.max_iterations)

                    trace_path = runner.trace_path
                    if trace_path:
                        st.caption(f"Trace: {trace_path}")

                    pending_question = None
                    if runner.last_step:
                        pending_question = (runner.last_step.get("assistant_user_request") or {}).get("question")
                    if not pending_question and trace_path:
                        payload = _safe_read_json(trace_path)
                        if payload and payload.get("steps"):
                            pending_question = (
                                (payload["steps"][-1].get("assistant_user_request") or {}).get("question")
                            )
                    if pending_question:
                        st.warning(f"Agent asked: {pending_question}")
                        allow_skip = st.checkbox(
                            "Proceed without responding",
                            key="interactive_skip_question",
                        )
                    else:
                        allow_skip = False

                    user_message = st.text_area(
                        "User follow-up (optional)",
                        key="interactive_user_message",
                        height=90,
                        placeholder="Ask a clarifying question or provide guidance...",
                    )

                    c1, c2, c3 = st.columns(3)
                    send_disabled = runner.done or (pending_question and not user_message.strip() and not allow_skip)
                    if c1.button("Send & Step", type="primary", disabled=send_disabled, key="interactive_step_send"):
                        with st.spinner("Running step..."):
                            try:
                                runner.step(user_message.strip() or None)
                            except Exception as e:
                                st.error(f"Step failed: {e}")
                        st.session_state["interactive_user_message"] = ""
                        st.rerun()

                    step_disabled = runner.done or (pending_question and not allow_skip)
                    if c2.button("Step (no user message)", disabled=step_disabled, key="interactive_step_plain"):
                        with st.spinner("Running step..."):
                            try:
                                runner.step(None)
                            except Exception as e:
                                st.error(f"Step failed: {e}")
                        st.rerun()

                    if c3.button("Stop run", key="interactive_stop"):
                        runner.stop()
                        st.session_state["interactive_runner"] = None
                        st.session_state["interactive_trace_path"] = ""
                        st.rerun()

                    trace_payload = _safe_read_json(trace_path) if trace_path else None
                    if trace_payload:
                        steps_live = trace_payload.get("steps") or []
                        if steps_live:
                            last_step = steps_live[-1]
                            st.markdown("#### Latest Step")
                            col_img, col_detail = st.columns([1, 1])
                            with col_img:
                                obs = last_step.get("observation") or {}
                                obs_path = obs.get("screenshot_path")
                                if _img_exists(obs_path):
                                    st.image(str(obs_path), use_container_width=True)
                                else:
                                    st.caption("No screenshot")
                            with col_detail:
                                user_extra = last_step.get("user_extra")
                                if user_extra:
                                    st.info(f"User: {user_extra}")
                                user_request = (last_step.get("assistant_user_request") or {}).get("question")
                                if user_request:
                                    st.warning(f"Agent asked: {user_request}")
                                assistant_text = (last_step.get("assistant") or {}).get("text") or ""
                                if assistant_text:
                                    st.text_area(
                                        "Assistant",
                                        assistant_text,
                                        height=140,
                                        disabled=True,
                                        label_visibility="collapsed",
                                    )
                                tool_uses = (last_step.get("assistant") or {}).get("tool_uses") or []
                                if tool_uses:
                                    actions = []
                                    for tu in tool_uses:
                                        action = (tu.get("input") or {}).get("action", "?")
                                        actions.append(action)
                                    st.caption(f"Actions: {', '.join(actions)}")
                                step_hash = last_step.get("prompt_hash")
                                if step_hash:
                                    st.caption(f"Prompt hash: {step_hash}")

                        with st.expander("Step timeline"):
                            rows = []
                            for s in steps_live[-12:]:
                                tool_uses = (s.get("assistant") or {}).get("tool_uses") or []
                                actions = [str((tu.get("input") or {}).get("action", "?")) for tu in tool_uses]
                                assistant_text = (s.get("assistant") or {}).get("text") or ""
                                ask_user = (s.get("assistant_user_request") or {}).get("question") or ""
                                rows.append({
                                    "step": s.get("step"),
                                    "user_extra": s.get("user_extra") or "",
                                    "actions": ", ".join(actions),
                                    "assistant": assistant_text[:120],
                                    "ask_user": ask_user[:120],
                                })
                            if rows:
                                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                            else:
                                st.caption("No steps yet.")

                else:
                    if st.button(
                        "Start interactive run",
                        type="primary",
                        key="interactive_start",
                        disabled=not interactive_ready,
                    ):
                        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_condition = (
                            condition_name.replace(" ", "_")
                            .replace("(", "")
                            .replace(")", "")
                            .replace("/", "_")
                        )
                        debug_dir = DEFAULT_DEBUG_DIR / safe_condition / task_id / f"run_01_{run_timestamp}"
                        prompt_hash = hashlib.sha256(assembled_prompt_effective.encode("utf-8")).hexdigest()
                        trace_meta = {
                            "run_timestamp": run_timestamp,
                            "condition_name": condition_name,
                            "task_id": task_id,
                            "task_instruction": instruction,
                            "target_url": target_url,
                            "api_url": api_url,
                            "discoverability": discoverability,
                            "capability": capability,
                            "model": ComputerUseAgent.SUPPORTED_MODELS.get(model, model),
                            "anthropic_beta": "computer-use-2025-01-24,prompt-caching-2024-07-31",
                            "display_width": DISPLAY_WIDTH,
                            "display_height": DISPLAY_HEIGHT,
                            "max_iterations": MAX_ITERATIONS,
                            "system_prompt": assembled_prompt_effective,
                            "interactive": True,
                        }
                        if selected_guidance_packs:
                            trace_meta.update({
                                "base_system_prompt": base_prompt_preview,
                                "guidance_packs": selected_guidance_packs,
                                "guidance_fragments": guidance_fragments,
                                "guidance_prompt_hash": prompt_hash,
                            })
                        if interactive_mode:
                            trace_meta.update({
                                "interactive_user_request": True,
                                "interactive_request_tag": ASK_USER_TAG,
                            })

                        runner = InteractiveRunner()
                        try:
                            runner.start(
                                instruction=instruction,
                                verifier=verifier,
                                condition_name=condition_name,
                                target_url=target_url,
                                api_url=api_url,
                                discoverability=discoverability,
                                capability=capability,
                            system_prompt=assembled_prompt_effective,
                                model=model,
                                debug_dir=debug_dir,
                                trace_meta=trace_meta,
                            )
                        except Exception as e:
                            st.error(f"Failed to start interactive run: {e}")
                            try:
                                runner.stop()
                            except Exception:
                                pass
                            st.session_state["interactive_runner"] = None
                        else:
                            st.session_state["interactive_runner"] = runner
                            st.session_state["interactive_trace_path"] = str(debug_dir / "trace.json")
                            st.session_state["interactive_user_message"] = ""
                            st.rerun()

    # === Packs & Guidance (Tab 1) ===
    with tabs[1]:
        st.subheader("Packs & Guidance")
        st.caption("Packs are opt-in extensions that add evaluators, fuzzers, guidance fragments, and advisors.")

        st.markdown("### What changes when you enable packs")
        st.markdown(
            "- Guidance packs append fragments to the system prompt and are recorded in trace meta.\n"
            "- Evaluator packs run after a trace is produced and write `eval_results.json` with pass/fail/uncertain decisions.\n"
            "- Fuzzer packs generate stress cases used by `flow_fuzz.py` when selected.\n"
            "- Advisor packs (optional LLM) propose guidance patches when evaluators fail."
        )

        st.markdown("### Outputs and artifacts")
        st.markdown(
            "- Trace meta includes `guidance_packs`, `guidance_fragments`, and `guidance_prompt_hash` when guidance is enabled.\n"
            "- Evaluations write `eval_results.json` next to the trace.\n"
            "- Advisors (if enabled) can write `guidance_patch.json`."
        )

        st.markdown("### Available packs in this workspace")
        try:
            registry = PackRegistry.discover()
            packs = registry.list_packs()
            pack_rows = []
            component_rows = []
            for pack in packs:
                pack_rows.append({
                    "pack_id": pack.id,
                    "name": pack.name,
                    "version": pack.version,
                    "description": pack.description,
                })
                for spec in pack.evaluators:
                    component_rows.append({
                        "pack_id": pack.id,
                        "type": "evaluator",
                        "id": spec.id,
                        "kind": spec.kind,
                        "severity": spec.severity,
                        "description": spec.description,
                    })
                for spec in pack.fuzzers:
                    component_rows.append({
                        "pack_id": pack.id,
                        "type": "fuzzer",
                        "id": spec.id,
                        "kind": spec.kind,
                        "severity": "",
                        "description": spec.description,
                    })
                for spec in pack.guidance:
                    component_rows.append({
                        "pack_id": pack.id,
                        "type": "guidance",
                        "id": spec.id,
                        "kind": spec.kind,
                        "severity": "",
                        "description": spec.description,
                    })
                for spec in pack.advisors:
                    component_rows.append({
                        "pack_id": pack.id,
                        "type": "advisor",
                        "id": spec.id,
                        "kind": spec.kind,
                        "severity": "",
                        "description": spec.description,
                    })

            if pack_rows:
                st.dataframe(pd.DataFrame(pack_rows), use_container_width=True)
            if component_rows:
                st.markdown("#### Pack components")
                st.dataframe(pd.DataFrame(component_rows), use_container_width=True)
        except Exception as e:
            st.warning(f"Failed to load packs: {e}")

        st.markdown("### How to enable")
        st.markdown(
            "- Use the Run Launcher to select Guidance Packs and Eval Packs.\n"
            "- In Trace Viewer, run evaluators and optionally request an advisor patch."
        )
        st.markdown("### CLI examples")
        st.code(
            "python benchmark_computeruse.py --guidance-packs guidance_basics --eval-packs commerce_safety\n"
            "python flow_fuzz.py --fuzz-pack basic_flow_fuzz --fuzzer basic_strategies --turns 3,4,5\n"
            "python pack_cli.py list-components --type all\n"
            "python pack_cli.py eval-trace --trace debug_screenshots/.../trace.json --eval-packs commerce_safety",
            language="bash",
        )

    # === Trace Viewer (Tab 2) ===
    with tabs[2]:
        if not traces:
            st.info("No traces available. Run a benchmark first.")
        elif not steps:
            st.info("Selected trace has no steps.")
        else:
            total_steps = len(steps)
            parent_trace_id = trace.get("parent_trace_id")
            branch_point_step = trace.get("branch_point_step")
            intervention_data = trace.get("intervention")
            registry = None
            packs = []
            try:
                registry = PackRegistry.discover()
                packs = registry.list_packs()
            except Exception:
                registry = None

            # === Header: minimap + info ===
            injection_steps = [i+1 for i, s in enumerate(steps) if s.get("user_extra")]
            step_markers = []
            for i, s in enumerate(steps):
                n = i + 1
                if s.get("user_extra"):
                    step_markers.append("💉")
                elif parent_trace_id and branch_point_step and n == branch_point_step:
                    step_markers.append("🎯")
                elif parent_trace_id and branch_point_step and n < branch_point_step:
                    step_markers.append("↩")
                else:
                    step_markers.append("·")

            info = f"{total_steps} steps: {''.join(step_markers)}"
            if parent_trace_id:
                info += f" | Branch@{branch_point_step} from {parent_trace_id[:8]}"
            if injection_steps:
                info += f" | 💉{injection_steps}"
            st.caption(info)

            # === Step slider ===
            step_idx = st.slider("Step", 1, total_steps, 1, key="trace_step") - 1
            step = steps[step_idx]

            assembled_prompt = meta.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
            base_prompt = meta.get("base_system_prompt") or assembled_prompt
            guidance_packs = meta.get("guidance_packs") or []
            guidance_fragments = meta.get("guidance_fragments") or []
            prompt_hash = meta.get("guidance_prompt_hash") or hashlib.sha256(
                assembled_prompt.encode("utf-8")
            ).hexdigest()

            # === Two-column layout: Screenshot | Details ===
            col_img, col_detail = st.columns([1, 1])

            with col_img:
                obs = step.get("observation") or {}
                obs_path = obs.get("screenshot_path")
                if _img_exists(obs_path):
                    st.image(str(obs_path), use_container_width=True)
                else:
                    st.caption("No screenshot")

            with col_detail:
                # Injection (if any)
                user_extra = step.get("user_extra")
                if user_extra:
                    st.error(f"💉 {user_extra}")

                # Actions
                tool_uses = (step.get("assistant") or {}).get("tool_uses") or []
                if tool_uses:
                    for tu in tool_uses:
                        inp = tu.get("input") or {}
                        action = inp.get("action", "?")
                        coord = inp.get("coordinate")
                        text = inp.get("text", "")
                        if coord:
                            st.code(f"{action} @ {coord}", language=None)
                        elif text:
                            st.code(f"{action}: {text[:50]}", language=None)
                        else:
                            st.code(action, language=None)

                # Reasoning
                assistant_text = (step.get("assistant") or {}).get("text") or ""
                if assistant_text:
                    st.text_area("Reasoning", assistant_text, height=150, disabled=True, label_visibility="collapsed")

                step_prompt_hash = step.get("prompt_hash") or prompt_hash
                if step_prompt_hash:
                    st.caption(f"Prompt hash: {step_prompt_hash}")

            # === Post-action screenshots (if multiple actions) ===
            tool_results = step.get("tool_results") or []
            if len(tool_results) > 1:
                st.caption("Post-action screenshots:")
                cols = st.columns(min(len(tool_results), 4))
                for i, tr in enumerate(tool_results):
                    imgp = tr.get("screenshot_path")
                    if _img_exists(imgp):
                        cols[i % 4].image(str(imgp), caption=tr.get('result_text', '')[:20])

            # === Investigation ===
            st.markdown("### Investigation")
            final_state = meta.get("final_state") or {}
            success_flag = meta.get("success")
            stage = final_state.get("stage") or "unknown"
            failure_reason = _derive_failure_reason(final_state, success_flag)
            progress = final_state.get("progress") or {}
            events = final_state.get("events") or {}
            event_counts = events.get("counts_by_type") or {}
            last_events = events.get("last_n") or []
            last_event_type = last_events[-1].get("type") if last_events else None

            actions = _extract_tool_actions(steps)
            action_tail = actions[-5:] if actions else []
            action_counts = _action_counts(actions) if actions else {}
            loop_info = _detect_action_loop(actions)
            task_complete = _task_complete_mentioned(steps)

            success_label = "unknown" if success_flag is None else ("yes" if success_flag else "no")
            cols = st.columns(4)
            cols[0].metric("Success", success_label)
            cols[1].metric("Stage", stage)
            cols[2].metric("Failure reason", failure_reason)
            cols[3].metric("Last event", last_event_type or "none")

            missing_events: list[str] = []
            if progress:
                if not progress.get("has_cart_items"):
                    missing_events.append("ADD_TO_CART")
                if not progress.get("checkout_started"):
                    missing_events.append("START_CHECKOUT")
                if not progress.get("order_completed"):
                    missing_events.append("CHECKOUT_COMPLETE")

            if missing_events:
                st.caption(f"Missing events: {', '.join(missing_events)}")

            if meta.get("variant_seed") is not None or meta.get("variant_level") is not None:
                st.caption(f"Variant seed: {meta.get('variant_seed')} | Level: {meta.get('variant_level')}")

            if event_counts:
                st.markdown("#### Event counts")
                st.dataframe(
                    pd.DataFrame([{"event": k, "count": v} for k, v in event_counts.items()]),
                    use_container_width=True,
                )

            if last_events:
                st.markdown("#### Recent events")
                st.dataframe(
                    pd.DataFrame(last_events[-8:]),
                    use_container_width=True,
                )

            if action_counts:
                st.markdown("#### Action mix")
                st.dataframe(
                    pd.DataFrame([{"action": k, "count": v} for k, v in action_counts.items()]),
                    use_container_width=True,
                )

            if action_tail:
                st.markdown("#### Last actions")
                lines = []
                for action in action_tail:
                    base = f"Step {action['step']:02d}: {action['action']}"
                    if action.get("coordinate"):
                        base += f" @ {action['coordinate']}"
                    if action.get("text"):
                        base += f" | {action['text']}"
                    lines.append(base)
                st.code("\n".join(lines), language=None)

            if loop_info:
                msg = f"Loop detected: {loop_info['action']} repeated {loop_info['length']}x (steps {loop_info['start_step']}-{loop_info['end_step']})."
                if loop_info.get("coordinate"):
                    msg += f" Coord: {list(loop_info['coordinate'])}"
                st.warning(msg)

            if task_complete and success_flag is False:
                st.warning("Agent declared TASK_COMPLETE but run did not succeed.")

            if not final_state:
                st.caption("Final state not recorded in this trace. Rerun with debug traces to enable deeper diagnosis.")

            # Suggested interventions
            hints = []
            hint_map = {
                "no_items_added": "No items added. Consider adding a clarification step or stronger product-finding guidance.",
                "checkout_not_started": "Items in cart but checkout never started. Add guidance to proceed to checkout once cart is correct.",
                "checkout_incomplete": "Checkout started but not completed. Check form filling or decoy button confusion.",
                "verifier_failed": "Order completed but verifier failed. Likely wrong item/variant/quantity.",
                "state_error": "State fetch failed. Check server stability or retry logic.",
            }
            hint = hint_map.get(failure_reason)
            if hint:
                hints.append(hint)
            if loop_info:
                hints.append("Looping behavior detected. Try a guidance fragment like “if an action fails twice, choose a different path.”")
            if not actions:
                hints.append("No tool actions recorded. Verify tool permissions and model/tool configuration.")
            if hints:
                st.markdown("#### Suggested interventions")
                for item in hints:
                    st.write(f"- {item}")

            # === LLM Diagnosis ===
            st.markdown("### LLM Diagnosis (optional)")
            diag_available = False
            if registry:
                for pack in packs:
                    if pack.id != "llm_judge_demo":
                        continue
                    for spec in pack.evaluators:
                        if spec.id == "failure_diagnosis":
                            diag_available = True
                            break
            if not diag_available:
                st.caption("LLM diagnosis evaluator not available. Enable pack `llm_judge_demo` and set ANTHROPIC_API_KEY.")
            else:
                default_rubric = (
                    "Identify likely failure modes and whether this run should be marked as fail or uncertain. "
                    "Include one concrete intervention the developer can try."
                )
                rubric = st.text_area("Diagnosis rubric", value=default_rubric, key="diag_rubric", height=80)
                judge_system_prompt = st.text_area(
                    "Judge system prompt (policies)",
                    key="diag_system_prompt",
                    height=100,
                    help="Use this to encode company policies, tone, refund rules, or safety boundaries.",
                )
                if st.button("Run LLM Diagnosis", key="run_llm_diag"):
                    if registry is None:
                        st.warning("Pack registry unavailable.")
                    else:
                        try:
                            results = run_evaluators_on_trace(
                                trace,
                                selected_evaluators=["llm_judge_demo:failure_diagnosis"],
                                config_overrides={
                                    "llm_judge_demo:failure_diagnosis": {
                                        "rubric": rubric,
                                        "system_prompt": judge_system_prompt,
                                    }
                                },
                                registry=registry,
                            )
                            if judge_system_prompt and trace_item:
                                _persist_judge_prompt(
                                    trace_item.path,
                                    trace,
                                    judge_system_prompt,
                                    "llm_judge_demo:failure_diagnosis",
                                )
                            if results:
                                st.session_state["llm_diag"] = results[0].__dict__
                                st.session_state["llm_diag_trace"] = str(trace_item.path)
                        except Exception as e:
                            st.error(f"LLM diagnosis failed: {e}")

                diag = st.session_state.get("llm_diag")
                diag_trace = st.session_state.get("llm_diag_trace")
                if diag and diag_trace == str(trace_item.path):
                    st.caption("Advisory only. Use alongside deterministic signals.")
                    st.markdown(f"**Decision:** {diag.get('decision')}")
                    if diag.get("confidence") is not None:
                        st.markdown(f"**Confidence:** {diag.get('confidence')}")
                    if diag.get("summary"):
                        st.markdown("**Summary**")
                        st.write(diag.get("summary"))
                    metrics = diag.get("metrics") or {}
                    if metrics.get("failure_reason") or metrics.get("suggested_intervention"):
                        st.markdown("**Diagnosis**")
                        if metrics.get("failure_reason"):
                            st.write(f"Failure reason: {metrics.get('failure_reason')}")
                        if metrics.get("suggested_intervention"):
                            st.write(f"Suggested intervention: {metrics.get('suggested_intervention')}")
                    evidence = diag.get("evidence") or []
                    if evidence:
                        st.markdown("**Evidence**")
                        for ev in evidence:
                            if isinstance(ev, dict):
                                step = ev.get("step_index")
                                note = ev.get("note")
                                if step is not None:
                                    st.write(f"Step {step}: {note or ''}")
                            else:
                                st.write(str(ev))

            # === Prompt & Guidance ===
            st.markdown("### Prompt")
            st.caption(f"Guidance packs: {', '.join(guidance_packs) if guidance_packs else 'None'}")
            st.caption(f"Prompt hash: {prompt_hash}")

            with st.expander("Assembled system prompt"):
                st.text_area("assembled_prompt", assembled_prompt, height=220, disabled=True, label_visibility="collapsed")

            with st.expander("Base system prompt"):
                st.text_area("base_prompt", base_prompt, height=220, disabled=True, label_visibility="collapsed")

            with st.expander("Guidance fragments"):
                if guidance_fragments:
                    for frag in guidance_fragments:
                        st.write(f"- {frag}")
                else:
                    st.caption("No guidance fragments on this trace.")

            with st.expander("Prompt diff (base → assembled)"):
                diff_lines = list(
                    difflib.unified_diff(
                        base_prompt.splitlines(),
                        assembled_prompt.splitlines(),
                        fromfile="base",
                        tofile="assembled",
                        lineterm="",
                    )
                )
                if diff_lines:
                    st.code("\n".join(diff_lines), language="diff")
                else:
                    st.caption("No differences.")

            # === Evaluators ===
            st.markdown("### Evaluators")
            st.caption("Deterministic checks over the trace. Gate fails if any error-level evaluator fails.")
            try:
                if registry is None:
                    registry = PackRegistry.discover()
                    packs = registry.list_packs()
                pack_ids = [p.id for p in packs if p.evaluators]
            except Exception as e:
                st.error(f"Failed to load packs: {e}")
                pack_ids = []
                packs = []
                registry = None

            judge_prompt = st.text_area(
                "Judge system prompt (policies)",
                key="eval_judge_prompt",
                height=100,
                help="Applied to all LLM evaluators unless a specific evaluator overrides system_prompt.",
            )

            selected_packs = st.multiselect("Packs", options=pack_ids, default=[])
            catalog_rows = []
            for pack in packs:
                if selected_packs and pack.id not in selected_packs:
                    continue
                for spec in pack.evaluators:
                    catalog_rows.append({
                        "pack": pack.id,
                        "evaluator": spec.id,
                        "severity": spec.severity,
                        "description": spec.description,
                        "default_config": spec.default_config,
                    })
            if catalog_rows:
                st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True)
            else:
                st.caption("No evaluators available for the current selection.")

            if st.button("Run Evaluators", type="primary"):
                if not selected_packs:
                    st.warning("Select at least one pack.")
                elif registry is None:
                    st.warning("No pack registry available.")
                else:
                    with st.spinner("Running evaluators..."):
                        results = run_evaluators_on_trace(
                            trace,
                            selected_packs=selected_packs,
                            registry=registry,
                            judge_system_prompt=judge_prompt,
                        )
                    if judge_prompt and trace_item:
                        _persist_judge_prompt(trace_item.path, trace, judge_prompt, "global")
                    st.session_state["eval_results"] = results
                    st.session_state["eval_trace_path"] = str(trace_item.path) if trace_item else ""
                    st.session_state["eval_packs"] = list(selected_packs)

            if st.session_state.get("eval_trace_path") == str(trace_item.path):
                results = st.session_state.get("eval_results") or []
                if results:
                    gate_failed = not gate_decision(results)
                    total = len(results)
                    failed = sum(1 for r in results if r.decision == "fail")
                    uncertain = sum(1 for r in results if r.decision == "uncertain")
                    error_failed = sum(1 for r in results if r.decision == "fail" and r.severity == "error")
                    warning_failed = sum(1 for r in results if r.decision == "fail" and r.severity != "error")

                    st.markdown("#### Results")
                    cols = st.columns(4)
                    cols[0].metric("Total", total)
                    cols[1].metric("Failed", failed)
                    cols[2].metric("Uncertain", uncertain)
                    cols[3].metric("Error Failed", error_failed)
                    st.caption(f"Gate failed: {gate_failed} | Warning fails: {warning_failed}")

                    table_rows = []
                    for res in results:
                        table_rows.append({
                            "pack": res.pack_id,
                            "evaluator": res.evaluator_id,
                            "decision": res.decision,
                            "severity": res.severity,
                            "confidence": res.confidence,
                            "summary": res.summary,
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                    for res in results:
                        status = str(res.decision).upper()
                        label = f"{res.pack_id}:{res.evaluator_id} [{res.severity}] {status}"
                        with st.expander(label, expanded=res.decision != "pass"):
                            if res.summary:
                                st.write(res.summary)
                            if res.confidence is not None:
                                try:
                                    st.write(f"Confidence: {float(res.confidence):.2f}")
                                except Exception:
                                    st.write(f"Confidence: {res.confidence}")
                            if res.metrics:
                                st.json(res.metrics)
                            if res.evidence:
                                st.markdown("**Evidence**")
                                for ev in res.evidence:
                                    if isinstance(ev, dict):
                                        step = ev.get("step_index")
                                        if step is not None:
                                            st.write(f"Step {step}")
                                        screenshot_path = ev.get("screenshot_path")
                                        if _img_exists(screenshot_path):
                                            st.image(str(screenshot_path), use_container_width=True)
                                        elif screenshot_path:
                                            st.caption(str(screenshot_path))
                                        tool_use_id = ev.get("tool_use_id")
                                        if tool_use_id:
                                            st.caption(f"tool_use_id: {tool_use_id}")
                                        note = ev.get("note")
                                        if note:
                                            st.caption(note)
                                    else:
                                        st.write(str(ev))

            # === Guidance Advisor ===
            st.markdown("### Guidance Advisor")
            if registry is None:
                st.caption("Pack registry unavailable.")
            else:
                advisor_pack_ids = [p.id for p in packs if p.advisors]
                if not advisor_pack_ids:
                    st.caption("No advisor packs available.")
                else:
                    advisor_pack = st.selectbox("Advisor Pack", advisor_pack_ids, key="advisor_pack_sel")
                    advisor_specs = registry.get_manifest(advisor_pack).advisors
                    advisor_ids = [s.id for s in advisor_specs]
                    advisor_id = st.selectbox("Advisor", advisor_ids, key="advisor_id_sel")

                    if st.button("Suggest Guidance Patch"):
                        results = st.session_state.get("eval_results") or []
                        if not results:
                            st.warning("Run evaluators first to generate context.")
                        else:
                            try:
                                advisor_spec = registry.get_advisor_spec(advisor_pack, advisor_id)
                                advisor_fn = registry.load_advisor(advisor_pack, advisor_id)
                                patch = suggest_guidance_patch(
                                    advisor_fn,
                                    trace,
                                    results,
                                    config=advisor_spec.default_config,
                                )
                                if patch is None:
                                    st.warning("Advisor skipped or returned no patch.")
                                else:
                                    st.session_state["guidance_patch"] = patch.__dict__
                                    st.session_state["guidance_patch_trace"] = str(trace_item.path)
                                    st.success("Guidance patch generated.")
                            except Exception as e:
                                st.error(f"Advisor failed: {e}")

                    patch = st.session_state.get("guidance_patch")
                    patch_trace = st.session_state.get("guidance_patch_trace")
                    if patch and patch_trace == str(trace_item.path):
                        st.markdown("#### Patch Preview")
                        st.json(patch)

                        base_prompt = meta.get("base_system_prompt") or meta.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
                        current_fragments = meta.get("guidance_fragments") or []
                        new_fragments = list(current_fragments) + (patch.get("new_fragments") or [])
                        patched_prompt = assemble_system_prompt(base_prompt, new_fragments)

                        branch_step = st.slider(
                            "Branch step for patch",
                            1,
                            total_steps,
                            min(step_idx + 1, total_steps),
                            key="guidance_branch_step",
                        )
                        headless = st.checkbox("Headless branch run", True, key="guidance_branch_headless")

                        if st.button("Apply Guidance Patch as Branch", type="primary"):
                            trees_dir = debug_root_p / "trees"
                            trees_dir.mkdir(parents=True, exist_ok=True)
                            trace_id = trace.get("trace_id") or meta.get("trace_id")
                            tree = _find_tree_for_trace(trace_id, trees_dir)
                            if not tree:
                                try:
                                    tree = TraceTree.create_from_existing_trace(trace_item.path, trees_dir)
                                    trace_id = tree.get_root().trace_id
                                except Exception as e:
                                    st.error(f"Failed to create trace tree: {e}")
                                    tree = None
                            if tree:
                                intervention = Intervention(
                                    type=InterventionType.GUIDANCE_PATCH,
                                    system_prompt=patched_prompt,
                                    guidance_fragments=new_fragments,
                                    guidance_patch=patch,
                                )
                                config = BranchExecutionConfig(
                                    tree=tree,
                                    parent_trace_id=trace_id,
                                    branch_point_step=branch_step,
                                    intervention=intervention,
                                    headless=headless,
                                    label="Guidance patch",
                                )
                                progress = st.progress(0)
                                status = st.empty()
                                def upd(s, c, t):
                                    if t > 0:
                                        progress.progress(c/t)
                                    status.text(s)
                                result = run_branch_sync(config, progress_callback=upd)
                                if result.success:
                                    st.success(f"Created branch: {result.new_trace_id[:8]}")
                                else:
                                    st.error(result.error)

    # === Trace Trees (Tab 3) ===
    with tabs[3]:
        trees_dir = debug_root_p / "trees"
        available_trees = TraceTree.list_trees(trees_dir) if trees_dir.exists() else []

        # Import row
        if traces:
            col_import, col_btn = st.columns([4, 1])
            with col_import:
                import_choice = st.selectbox(
                    "Import trace as tree",
                    options=[None] + list(range(len(traces))),
                    format_func=lambda i: "Select trace..." if i is None else traces[i].label[:50],
                    key="import_trace_select"
                )
            with col_btn:
                st.write("")
                if import_choice is not None and st.button("Import", key="import_btn"):
                    try:
                        new_tree = TraceTree.create_from_existing_trace(traces[import_choice].path, trees_dir)
                        st.success(f"Created: {new_tree.tree_id[:8]}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if available_trees:
            # Tree selector
            tree_options = {t["tree_id"]: f"{t['description'][:30]} ({t['trace_count']})" for t in available_trees}
            selected_tree_id = st.selectbox("Tree", list(tree_options.keys()), format_func=lambda x: tree_options[x], key="tree_select")

            if selected_tree_id:
                try:
                    tree = TraceTree(selected_tree_id, trees_dir)

                    # Tree visualization
                    selected_node_id = render_trace_tree(tree)
                    if selected_node_id:
                        st.session_state["selected_trace_node"] = selected_node_id

                    current_node_id = st.session_state.get("selected_trace_node", tree.get_root().trace_id)
                    current_node = tree.get_node(current_node_id)

                    # Node info line
                    node_info = f"**{current_node_id[:8]}** · {current_node.steps_count} steps"
                    if current_node.parent_trace_id:
                        node_info += f" · branched from {current_node.parent_trace_id[:8]}@{current_node.branch_point_step}"
                    if current_node.success is not None:
                        node_info += f" · {'✓ Success' if current_node.success else '✗ Failed'}"
                    st.markdown(node_info)

                    if current_node.parent_trace_id:
                        try:
                            parent_steps = tree.get_full_step_sequence(current_node.parent_trace_id)
                            current_steps = tree.get_full_step_sequence(current_node.trace_id)
                            divergence = _first_divergence_step(parent_steps, current_steps)
                            if divergence:
                                st.caption(f"First divergence step: {divergence}")
                        except Exception:
                            pass

                    # Load trace for branch creation
                    try:
                        node_trace = tree.load_trace(current_node_id)
                        node_steps = node_trace.get("steps", [])
                        max_step = len(node_steps)

                        if max_step > 0:
                            st.divider()

                            # === Branch Creation with Step Preview ===
                            col_left, col_right = st.columns([1, 1])

                            with col_left:
                                st.markdown("**Create Branch**")
                                branch_step = st.slider("Branch at step", 1, max_step, min(3, max_step), key="branch_step")

                                intervention_type = st.radio("Intervention", ["Prompt", "Model", "Action"], horizontal=True, key="int_type")

                                intervention = None
                                if intervention_type == "Prompt":
                                    txt = st.text_area("Inject user message", height=80, key="prompt_txt",
                                                      placeholder="Actually, I changed my mind - get me the large size instead")
                                    if txt:
                                        intervention = Intervention(type=InterventionType.PROMPT_INSERT, prompt_text=txt)
                                elif intervention_type == "Model":
                                    m = st.selectbox("Switch to model", ["sonnet", "haiku"], key="model_sel")
                                    intervention = Intervention(type=InterventionType.MODEL_SWAP, model=ComputerUseAgent.SUPPORTED_MODELS[m])
                                else:
                                    act = st.selectbox("Force action", ["left_click", "type", "key"], key="act_type")
                                    if act == "left_click":
                                        cx, cy = st.columns(2)
                                        coords = [int(cx.number_input("X", 0, 1280, 640, key="cx")),
                                                  int(cy.number_input("Y", 0, 800, 400, key="cy"))]
                                        intervention = Intervention(type=InterventionType.TOOL_OVERRIDE,
                                                                   forced_action={"action": "left_click", "coordinate": coords})
                                    elif act == "type":
                                        t = st.text_input("Text to type", key="type_txt")
                                        if t:
                                            intervention = Intervention(type=InterventionType.TOOL_OVERRIDE,
                                                                       forced_action={"action": "type", "text": t})
                                    elif act == "key":
                                        k = st.text_input("Key to press", "Enter", key="key_txt")
                                        if k:
                                            intervention = Intervention(type=InterventionType.TOOL_OVERRIDE,
                                                                       forced_action={"action": "key", "key": k})

                                c1, c2 = st.columns([1, 2])
                                headless = c1.checkbox("Headless", True, key="headless")
                                if intervention and c2.button("Execute Branch", type="primary", key="exec_branch"):
                                    config = BranchExecutionConfig(tree=tree, parent_trace_id=current_node_id,
                                        branch_point_step=branch_step, intervention=intervention, headless=headless)
                                    progress = st.progress(0)
                                    status = st.empty()
                                    def upd(s, c, t):
                                        if t > 0: progress.progress(c/t)
                                        status.text(s)
                                    result = run_branch_sync(config, progress_callback=upd)
                                    if result.success:
                                        st.success(f"Created: {result.new_trace_id[:8]}")
                                        st.rerun()
                                    else:
                                        st.error(result.error)

                            with col_right:
                                # === Step Preview (what agent saw/did at this step) ===
                                st.markdown(f"**Step {branch_step} Preview** (intervention applies here)")
                                if branch_step <= len(node_steps):
                                    preview_step = node_steps[branch_step - 1]

                                    # Screenshot
                                    obs = preview_step.get("observation") or {}
                                    obs_path = obs.get("screenshot_path")
                                    if _img_exists(obs_path):
                                        st.image(str(obs_path), use_container_width=True)

                                    # What agent did
                                    tool_uses = (preview_step.get("assistant") or {}).get("tool_uses") or []
                                    if tool_uses:
                                        actions = []
                                        for tu in tool_uses:
                                            inp = tu.get("input") or {}
                                            a = inp.get("action", "?")
                                            if inp.get("coordinate"):
                                                actions.append(f"{a}@{inp['coordinate']}")
                                            elif inp.get("text"):
                                                actions.append(f"{a}: {inp['text'][:30]}")
                                            else:
                                                actions.append(a)
                                        st.code(" → ".join(actions), language=None)

                                    # Agent's reasoning
                                    reasoning = (preview_step.get("assistant") or {}).get("text") or ""
                                    if reasoning:
                                        st.text_area("Agent reasoning", reasoning, height=100, disabled=True, label_visibility="collapsed")

                    except Exception as e:
                        st.error(f"Error loading trace: {e}")

                except Exception as e:
                    st.error(str(e))

        elif not traces:
            st.info("Run a benchmark first to generate traces.")

    # === Reliability Dashboard (Tab 4) ===
    with tabs[4]:
        st.subheader("Reliability Dashboard")
        st.caption("Run seeded sweeps across UI variants and summarize failures and event coverage.")

        st.markdown("### Load Existing Report")
        reports_dir = Path("benchmark_results")
        report_files: list[Path] = []
        if reports_dir.exists():
            report_files = sorted(
                reports_dir.glob("reliability_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

        if report_files:
            report_labels = [
                f"{p.name} ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
                for p in report_files
            ]
            report_choice = st.selectbox(
                "Past reports",
                options=list(range(len(report_files))),
                format_func=lambda i: report_labels[i],
                key="rel_report_select",
            )
            if st.button("Load selected report", key="rel_load_report"):
                try:
                    chosen = report_files[int(report_choice)]
                    report = json.loads(chosen.read_text(encoding="utf-8"))
                    st.session_state["reliability_report"] = report
                    st.session_state["reliability_out_path"] = str(chosen)
                    st.success(f"Loaded report: {chosen}")
                except Exception as e:
                    st.error(f"Failed to load report: {e}")
        else:
            st.caption("No reliability reports found in benchmark_results/ yet.")

        col_left, col_right = st.columns(2)
        with col_left:
            rel_app = st.selectbox("App", ["treatment", "treatment-docs", "baseline"], index=0, key="rel_app")
            rel_start = st.selectbox("Start point", ["root", "agent"], index=0, key="rel_start")
            rel_discoverability = st.selectbox("Discoverability", ["navbar", "hidden"], index=0, key="rel_disc")
        with col_right:
            rel_capability = st.selectbox("Capability", ["advantage", "parity"], index=0, key="rel_cap")
            rel_model = st.selectbox("Model", ["sonnet", "haiku"], index=0, key="rel_model")
            task_options = {t["id"]: f"{t['id']}: {t['instruction'][:50]}..." for t in PREDEFINED_TASKS}
            rel_task_id = st.selectbox(
                "Task",
                options=list(task_options.keys()),
                format_func=lambda x: task_options[x],
                key="rel_task"
            )

        rel_seeds = st.text_input(
            "Seeds",
            value="0-9",
            help="Base sweep seeds. Examples: 0-9 or 0,1,2,3",
        )
        rel_variant_level = st.slider(
            "Variant level",
            0,
            3,
            0,
            key="rel_variant_level",
            help=(
                "0 = baseline UI. Higher levels add stronger perturbations (e.g., decoy button at >=2 "
                "and optional latency at >=3 when seed%3==0)."
            ),
        )
        rel_adaptive = st.checkbox(
            "Adaptive expansion",
            value=False,
            key="rel_adaptive",
            help=(
                "Rerun failing seeds at level+1 (capped at 3). Budget limits the number of extra runs."
            ),
        )
        rel_budget = st.number_input(
            "Adaptive budget",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            disabled=not rel_adaptive,
            key="rel_budget",
            help="Maximum number of adaptive reruns (each rerun uses the same seed with higher level)."
        )
        rel_prompt_override = st.text_area(
            "System prompt override (optional)",
            height=120,
            key="rel_prompt_override"
        )

        st.markdown("### Command")
        rel_cmd = [
            "python", "reliability_eval.py",
            "--app", rel_app,
            "--task", rel_task_id,
            "--start", rel_start,
            "--discoverability", rel_discoverability,
            "--capability", rel_capability,
            "--model", rel_model,
            "--seeds", rel_seeds,
            "--variant-level", str(int(rel_variant_level)),
        ]
        if rel_adaptive:
            rel_cmd += ["--adaptive", "--adaptive-budget", str(int(rel_budget))]
        st.code(" ".join(rel_cmd))

        if st.button("Run Reliability Eval", type="primary"):
            tmp_prompt_file = None
            cmd = list(rel_cmd)
            if rel_prompt_override.strip():
                tmp_dir = Path(".tmp_prompts")
                tmp_dir.mkdir(exist_ok=True)
                tmp_prompt_file = tmp_dir / f"reliability_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                tmp_prompt_file.write_text(rel_prompt_override, encoding="utf-8")
                cmd += ["--prompt-file", str(tmp_prompt_file)]

            out_path = Path(f"benchmark_results/reliability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            cmd += ["--out", str(out_path)]

            with st.spinner("Running reliability sweep…"):
                proc = subprocess.run(cmd, capture_output=True, text=True)

            st.markdown("**STDOUT**")
            st.text_area("rel_stdout", value=proc.stdout, height=200)
            st.markdown("**STDERR**")
            st.text_area("rel_stderr", value=proc.stderr, height=200)

            if proc.returncode != 0:
                st.error(f"Command failed with exit code {proc.returncode}")
            else:
                try:
                    report = json.loads(out_path.read_text(encoding="utf-8"))
                    st.session_state["reliability_report"] = report
                    st.session_state["reliability_out_path"] = str(out_path)
                    st.success(f"Report saved to {out_path}")
                except Exception as e:
                    st.error(f"Failed to read report: {e}")

        report = st.session_state.get("reliability_report")
        if report:
            summary = report.get("summary", {})
            expanded = report.get("expanded_summary") or {}

            base_rate = summary.get("success_rate", 0) * 100
            st.metric("Base success rate", f"{base_rate:.1f}%")

            if expanded.get("expanded_success_rate") is not None:
                expanded_rate = expanded.get("expanded_success_rate", 0) * 100
                st.metric("Expanded success rate", f"{expanded_rate:.1f}%")

            st.markdown("### Failure stage histogram (base)")
            st.caption("Stage meaning: browse=no cart items, cart=items but no START_CHECKOUT, checkout=started but no order, done=order completed but verifier failed.")
            base_hist = summary.get("failure_stage_histogram", {})
            if base_hist:
                st.bar_chart(pd.Series(base_hist))
            else:
                st.caption("No failures recorded in base runs.")

            base_reason = summary.get("failure_reason_histogram", {})
            if base_reason:
                st.markdown("### Failure reasons (base)")
                st.bar_chart(pd.Series(base_reason))

            if expanded.get("failure_stage_shift"):
                st.markdown("### Failure stage histogram (adaptive)")
                adaptive_hist = expanded.get("failure_stage_shift", {}).get("adaptive") or {}
                if adaptive_hist:
                    st.bar_chart(pd.Series(adaptive_hist))
                else:
                    st.caption("No failures recorded in adaptive runs.")

            adaptive_reason = expanded.get("failure_reason_shift", {}).get("adaptive") if expanded else None
            if adaptive_reason:
                st.markdown("### Failure reasons (adaptive)")
                st.bar_chart(pd.Series(adaptive_reason))

            st.markdown("### Runs")
            base_runs = report.get("base_runs", [])
            adaptive_runs = report.get("adaptive_runs", [])
            run_rows = base_runs + adaptive_runs
            if run_rows:
                df = pd.DataFrame(run_rows)
                columns = [
                    c for c in [
                        "seed",
                        "variant_level",
                        "success",
                        "stage",
                        "failure_reason",
                        "last_event_type",
                        "cart_total_items",
                        "steps",
                        "events_total",
                        "trace_path",
                    ]
                    if c in df.columns
                ]
                if "error" in df.columns:
                    columns.append("error")
                st.dataframe(df[columns], use_container_width=True)
            else:
                st.caption("No runs to display.")

            if run_rows:
                st.markdown("### Run Inspector")
                run_labels: list[str] = []
                run_map: dict[str, dict] = {}
                for run in run_rows:
                    label = (
                        f\"{'adaptive' if run.get('is_adaptive') else 'base'} "
                        f\"seed {run.get('seed')} "
                        f\"L{run.get('variant_level')} "
                        f\"{'PASS' if run.get('success') else 'FAIL'} "
                        f\"stage={run.get('stage')} "
                        f\"reason={run.get('failure_reason')}\"
                    )
                    run_labels.append(label)
                    run_map[label] = run

                selected_label = st.selectbox("Inspect a run", options=run_labels, key="rel_inspect_run")
                selected_run = run_map.get(selected_label, {})

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Stage", selected_run.get("stage", "unknown"))
                col_b.metric("Failure reason", selected_run.get("failure_reason", "unknown"))
                col_c.metric("Last event", selected_run.get("last_event_type") or "none")

                missing = selected_run.get("missing_events") or []
                if missing:
                    st.caption(f\"Missing events: {', '.join(missing)}\")

                cart_preview = selected_run.get("cart_items_preview") or []
                if cart_preview:
                    st.markdown("**Cart preview**")
                    st.dataframe(pd.DataFrame(cart_preview), use_container_width=True)

                action_tail = selected_run.get("action_log_tail") or []
                if action_tail:
                    st.markdown("**Last actions**")
                    st.code(\"\\n\".join(action_tail), language=None)

                hint_map = {
                    "no_items_added": "No cart items detected. Suggest improving product discovery or adding a clarifying question when unsure.",
                    "checkout_not_started": "Items in cart but checkout not started. Add guidance like “proceed to checkout after items are correct.”",
                    "checkout_incomplete": "Checkout started but not completed. Check form filling and decoy button confusion.",
                    "verifier_failed": "Order completed but verifier failed. Likely wrong variant/quantity or incorrect item.",
                    "state_error": "State fetch failed. Ensure the app server is stable and reachable.",
                }
                hint = hint_map.get(selected_run.get("failure_reason"))
                if hint:
                    st.info(hint)

                trace_path = selected_run.get("trace_path")
                if trace_path and st.button("Open this trace in Viewer", key="rel_open_trace"):
                    st.session_state["trace_path_override"] = trace_path
                    st.success("Trace selected. Switch to Trace Viewer tab.")

            st.markdown("### Event counts (base)")
            event_counts = summary.get("event_coverage", {}).get("counts_by_type") or {}
            if event_counts:
                st.dataframe(pd.DataFrame([
                    {"event": k, "count": v} for k, v in event_counts.items()
                ]), use_container_width=True)
            else:
                st.caption("No events recorded.")

            trace_options = {}
            for run in run_rows:
                trace_path = run.get("trace_path")
                if trace_path:
                    label = f"{'adaptive' if run.get('is_adaptive') else 'base'} seed {run.get('seed')} (L{run.get('variant_level')})"
                    trace_options[label] = trace_path

            if trace_options:
                st.markdown("### Open Trace in Viewer")
                selection = st.selectbox("Trace", options=list(trace_options.keys()))
                if st.button("Open selected trace"):
                    st.session_state["trace_path_override"] = trace_options[selection]
                    st.success("Trace selected. Switch to Trace Viewer tab.")


if __name__ == "__main__":
    main()
