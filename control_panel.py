#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Control Panel (MVP) for Traces + Counterfactual Replay.

This is intentionally lightweight and interview-demo friendly.

Features
--------
1) Trace Browser + Viewer
   - Lists trace.json files under DEBUG_DIR (default: debug_screenshots/)
   - Renders per-step: observation screenshot, assistant text, tool uses/results

2) Counterfactual (Shadow) Replay
   - Runs replay_trace.shadow_counterfactual on a selected trace
   - Lets you edit the system prompt + pick model
   - Shows agreement rate + first divergence + per-step comparison table

3) Run Launcher (optional)
   - Generates (and optionally executes) benchmark/fuzz commands with chosen env filters

Usage
-----
    pip install -r requirements.txt
    pip install streamlit pandas pillow

    streamlit run control_panel.py

Notes
-----
- Shadow mode requires ANTHROPIC_API_KEY.
- Benchmark/fuzz runs require Playwright browsers installed:
    playwright install chromium
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

from replay_trace import _load_trace, shadow_counterfactual
from benchmark_computeruse import DEBUG_DIR as DEFAULT_DEBUG_DIR, SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT


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
        if not payload or payload.get("schema_version") != "trace.v1":
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


def _render_action_sig(sig: Optional[dict[str, Any]]) -> str:
    if not sig:
        return "(none)"
    action = sig.get("action") or "?"
    coord = sig.get("coordinate")
    text = sig.get("text")
    parts = [str(action)]
    if coord:
        parts.append(f"@{coord}")
    if text:
        t = str(text)
        parts.append(f"\"{t[:40]}\"" + ("…" if len(t) > 40 else ""))
    return " ".join(parts)


def main() -> None:
    st.set_page_config(page_title="AgentOps Control Panel (MVP)", layout="wide")
    st.title("AgentOps Control Panel (MVP)")

    # Sidebar config
    st.sidebar.header("Trace Source")
    debug_root = st.sidebar.text_input("Trace directory", value=str(DEFAULT_DEBUG_DIR))
    debug_root_p = Path(debug_root)

    traces = find_traces(debug_root_p)
    if not traces:
        st.sidebar.warning("No traces found. Run with DEBUG_SCREENSHOTS=1 to generate trace.json files.")
        st.stop()

    trace_labels = [t.label for t in traces]
    choice = st.sidebar.selectbox("Select a trace", options=list(range(len(traces))), format_func=lambda i: trace_labels[i])
    trace_item = traces[int(choice)]

    trace = _load_trace(trace_item.path)
    meta = trace.get("meta") or {}
    steps = trace.get("steps") or []

    st.subheader("Trace Meta")
    st.json(meta)

    tabs = st.tabs(["Trace Viewer", "Counterfactual (Shadow)", "Run Launcher"])

    # === Trace Viewer ===
    with tabs[0]:
        st.subheader("Step-by-step Viewer")
        if not steps:
            st.info("Trace has no steps.")
        else:
            step_idx = st.slider("Step", min_value=1, max_value=len(steps), value=1, step=1) - 1
            step = steps[step_idx]

            obs = step.get("observation") or {}
            obs_path = obs.get("screenshot_path")
            dbg_path = obs.get("debug_screenshot_path")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Observation (what the policy saw)**")
                if _img_exists(obs_path):
                    st.image(str(obs_path))
                else:
                    st.caption(f"No image at: {obs_path}")
            with col2:
                st.markdown("**Debug start-of-step screenshot (may differ)**")
                if _img_exists(dbg_path):
                    st.image(str(dbg_path))
                else:
                    st.caption(f"No image at: {dbg_path}")

            st.markdown("**Assistant text**")
            st.code((step.get("assistant") or {}).get("text") or "", language="markdown")

            # Tool uses
            tool_uses = (step.get("assistant") or {}).get("tool_uses") or []
            if tool_uses:
                st.markdown("**Tool uses**")
                df = pd.DataFrame([{
                    "idx": i + 1,
                    "action": (tu.get("input") or {}).get("action"),
                    "coordinate": (tu.get("input") or {}).get("coordinate"),
                    "text": (tu.get("input") or {}).get("text"),
                    "raw": json.dumps(tu.get("input") or {}, ensure_ascii=False),
                } for i, tu in enumerate(tool_uses)])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No tool_use blocks in this step.")

            # Tool results
            tool_results = step.get("tool_results") or []
            if tool_results:
                st.markdown("**Tool results (replayed)**")
                for i, tr in enumerate(tool_results, start=1):
                    st.markdown(f"*Result {i}* — {tr.get('result_text')}")
                    imgp = tr.get("screenshot_path")
                    if _img_exists(imgp):
                        st.image(str(imgp))
            else:
                st.caption("No tool_results recorded in this step.")

    # === Counterfactual ===
    with tabs[1]:
        st.subheader("Shadow Counterfactual Replay")
        st.caption("Runs a different model/prompt against the frozen observation trace (off-policy).")

        model_default = meta.get("model") or "claude-sonnet-4-5-20250929"
        model = st.text_input("Model", value=model_default)

        prompt_default = meta.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        system_prompt = st.text_area("System prompt", value=prompt_default, height=240)

        max_steps = st.number_input("Max steps", min_value=1, max_value=max(1, len(steps)), value=min(10, max(1, len(steps))))

        if st.button("Run counterfactual", type="primary"):
            if not os.getenv("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY is not set. Shadow mode requires it.")
            else:
                with st.spinner("Running shadow counterfactual…"):
                    res = __run_shadow(trace, model=model, system_prompt=system_prompt, max_steps=int(max_steps))

                st.success("Done.")
                st.json({k: res.get(k) for k in [
                    "task_id", "model", "steps_evaluated", "comparable_steps",
                    "first_action_agreement", "first_divergence_step"
                ]})

                # Per-step table
                rows = []
                for p, o in zip(res.get("predicted_actions") or [], res.get("original_actions") or []):
                    rows.append({
                        "step": p.get("step"),
                        "predicted": _render_action_sig(p.get("first_action")),
                        "original": _render_action_sig(o.get("first_action")),
                    })
                if rows:
                    st.markdown("**Per-step first-action comparison**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Save report next to trace
                out_path = trace_item.path.parent / f"counterfactual_{model.replace('/', '_')}.json"
                out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
                st.info(f"Saved report: {out_path}")

    # === Run Launcher ===
    with tabs[2]:
        st.subheader("Run Launcher")
        st.caption("Generates commands (and can optionally execute them) for benchmark/fuzz runs.")

        run_type = st.selectbox("Run type", options=["benchmark", "fuzz"], index=0)

        app = st.selectbox("App", options=["baseline", "treatment", "treatment-docs"], index=1)
        start = st.selectbox("Start", options=["root", "agent"], index=0)
        discoverability = st.selectbox("Discoverability", options=["navbar", "hidden"], index=0)
        capability = st.selectbox("Capability", options=["advantage", "parity"], index=0)

        tasks = st.text_input("Tasks (comma-separated, benchmark only)", value="all")
        runs_per_task = st.number_input("Runs per task", min_value=1, max_value=20, value=1)

        prompt_override = st.text_area("Optional prompt override (leave empty to use default)", value="", height=160)

        # Build command
        cmd: list[str] = ["python"]
        if run_type == "benchmark":
            cmd += ["benchmark_computeruse.py",
                    "--app", app,
                    "--start", start,
                    "--discoverability", discoverability,
                    "--capability", capability,
                    "--tasks", tasks,
                    "--runs-per-task", str(int(runs_per_task))]
        else:
            cmd += ["flow_fuzz.py",
                    "--app", app,
                    "--start", start,
                    "--discoverability", discoverability,
                    "--capability", capability]

        prompt_file = None
        if prompt_override.strip():
            tmp_dir = Path(".tmp_prompts")
            tmp_dir.mkdir(exist_ok=True)
            prompt_file = tmp_dir / f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            prompt_file.write_text(prompt_override, encoding="utf-8")
            cmd += ["--system-prompt-file", str(prompt_file)]

        st.markdown("**Command**")
        st.code(" ".join(cmd))

        if st.button("Run now (blocking)"):
            with st.spinner("Running… (this will block until completion)"):
                p = subprocess.run(cmd, capture_output=True, text=True)
            st.markdown("**STDOUT**")
            st.text_area("stdout", value=p.stdout, height=240)
            st.markdown("**STDERR**")
            st.text_area("stderr", value=p.stderr, height=240)
            if p.returncode != 0:
                st.error(f"Command failed with exit code {p.returncode}")
            else:
                st.success("Run completed.")

        if prompt_file:
            st.caption(f"Wrote prompt override to: {prompt_file}")


def __run_shadow(trace: dict[str, Any], model: str, system_prompt: str, max_steps: int) -> dict[str, Any]:
    # shadow_counterfactual is async; Streamlit is synchronous.
    import asyncio
    try:
        return asyncio.run(shadow_counterfactual(trace, model=model, system_prompt=system_prompt, max_steps=max_steps))
    except RuntimeError:
        # Fallback for environments with an active loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(shadow_counterfactual(trace, model=model, system_prompt=system_prompt, max_steps=max_steps))
        finally:
            loop.close()


if __name__ == "__main__":
    main()
