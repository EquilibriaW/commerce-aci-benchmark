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

import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import streamlit as st
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "\nThis control panel requires Streamlit.\n\n"
        "Install it with:\n"
        "  pip install streamlit pandas pillow\n\n"
        f"Import error: {e}\n"
    )

from replay_trace import _load_trace
from benchmark_computeruse import DEBUG_DIR as DEFAULT_DEBUG_DIR, TASKS as PREDEFINED_TASKS, ComputerUseAgent

# Import branching modules
from trace_tree import TraceTree, Intervention, InterventionType
from branch_executor import BranchExecutionConfig, run_branch_sync
from tree_visualization import render_trace_tree


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
        # Support both v1 and v2 schemas
        schema = payload.get("schema_version", "")
        if schema not in ("trace.v1", "trace.v2"):
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


def main() -> None:
    st.set_page_config(page_title="AgentOps Control Panel", layout="wide")
    st.title("AgentOps Control Panel")

    # Sidebar config
    st.sidebar.header("Trace Source")
    debug_root = st.sidebar.text_input("Trace directory", value=str(DEFAULT_DEBUG_DIR))
    debug_root_p = Path(debug_root)

    traces = find_traces(debug_root_p)

    # Always show tabs - Run Launcher doesn't require traces
    tabs = st.tabs(["Run Launcher", "Trace Viewer", "Trace Trees"])

    # Trace selection (only if traces exist)
    trace_item = None
    trace = None
    meta = {}
    steps = []

    if traces:
        trace_labels = [t.label for t in traces]
        choice = st.sidebar.selectbox("Select a trace", options=list(range(len(traces))), format_func=lambda i: trace_labels[i])
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

        if prompt_override.strip():
            tmp_dir = Path(".tmp_prompts")
            tmp_dir.mkdir(exist_ok=True)
            prompt_file = tmp_dir / f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            prompt_file.write_text(prompt_override, encoding="utf-8")
            cmd += ["--system-prompt-file", str(prompt_file)]

        # --- Show Command & Run Button ---
        st.markdown("### Command")
        st.code(" ".join(cmd))

        # Validation
        can_run = True
        if task_mode == "Custom Goal" and not custom_instruction.strip():
            can_run = False
            st.warning("Enter a goal/instruction to run.")
        elif task_mode == "Predefined Tasks" and not selected_task_ids:
            can_run = False

        if can_run and st.button("Run now", type="primary"):
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

    # === Trace Viewer (Tab 1) ===
    with tabs[1]:
        if not traces:
            st.info("No traces available. Run a benchmark first.")
        elif not steps:
            st.info("Selected trace has no steps.")
        else:
            total_steps = len(steps)
            parent_trace_id = trace.get("parent_trace_id")
            branch_point_step = trace.get("branch_point_step")
            intervention_data = trace.get("intervention")

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

            # === Post-action screenshots (if multiple actions) ===
            tool_results = step.get("tool_results") or []
            if len(tool_results) > 1:
                st.caption("Post-action screenshots:")
                cols = st.columns(min(len(tool_results), 4))
                for i, tr in enumerate(tool_results):
                    imgp = tr.get("screenshot_path")
                    if _img_exists(imgp):
                        cols[i % 4].image(str(imgp), caption=tr.get('result_text', '')[:20])

    # === Trace Trees (Tab 2) ===
    with tabs[2]:
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


if __name__ == "__main__":
    main()
