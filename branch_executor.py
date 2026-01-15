#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Branch Executor - Hybrid replay-then-live execution for trace branching.

This module handles the execution of trace branches:
1. Replay steps 1 to N-1 deterministically
2. Apply intervention at step N
3. Continue with live agent execution
4. Save the new branch trace

Usage:
    from branch_executor import BranchExecutor, BranchExecutionConfig

    config = BranchExecutionConfig(
        tree=tree,
        parent_trace_id="abc123",
        branch_point_step=5,
        intervention=Intervention(type=InterventionType.PROMPT_INSERT, prompt_text="..."),
    )

    executor = BranchExecutor(server_manager, progress_callback=lambda s,c,t: print(f"{s}: {c}/{t}"))
    result = await executor.execute_branch(config)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from anthropic import Anthropic
from playwright.async_api import async_playwright

from trace_tree import TraceTree, Intervention, InterventionType
from server_manager import ServerManager
from trace_rehydrate import build_conversation_from_steps


# Import from benchmark_computeruse - these are needed for execution
# We import lazily to avoid circular imports
def _get_benchmark_imports():
    from benchmark_computeruse import (
        ComputerUseAgent,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        SYSTEM_PROMPT,
        ANTHROPIC_API_KEY,
        TASKS,
    )
    return ComputerUseAgent, DISPLAY_WIDTH, DISPLAY_HEIGHT, SYSTEM_PROMPT, ANTHROPIC_API_KEY, TASKS


def trace_to_messages(trace: dict, up_to_step: int) -> list[dict]:
    """Rebuild an Anthropic message list from trace steps up to a step index."""
    meta = trace.get("meta") or {}
    instruction = meta.get("task_instruction") or "Continue the task"
    steps = trace.get("steps") or []
    if up_to_step is None:
        up_to_step = len(steps)
    up_to_step = max(0, min(int(up_to_step), len(steps)))
    trace_dir = trace.get("_trace_dir")
    return build_conversation_from_steps(steps[:up_to_step], instruction=instruction, trace_dir=trace_dir)


@dataclass
class BranchExecutionConfig:
    """Configuration for a branch execution."""
    tree: TraceTree
    parent_trace_id: str
    branch_point_step: int
    intervention: Intervention
    headless: bool = True
    max_iterations: int = 18
    label: str = ""
    rehydrate_conversation: bool = True


@dataclass
class BranchExecutionResult:
    """Result of a branch execution."""
    success: bool
    new_trace_id: str
    steps_count: int
    error: Optional[str] = None
    trace_path: Optional[Path] = None


class BranchExecutor:
    """Executes a branched trace: replay then live execution."""

    def __init__(
        self,
        server_manager: ServerManager,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """Initialize the executor.

        Args:
            server_manager: ServerManager instance for starting/stopping servers
            progress_callback: Optional callback for progress updates (status, current, total)
        """
        self.server_manager = server_manager
        self.progress_callback = progress_callback

    def _report_progress(self, status: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(status, current, total)

    async def execute_branch(self, config: BranchExecutionConfig) -> BranchExecutionResult:
        """Execute a branch from the given configuration.

        Args:
            config: Branch execution configuration

        Returns:
            BranchExecutionResult with success status and new trace info
        """
        # Import benchmark components
        ComputerUseAgent, DISPLAY_WIDTH, DISPLAY_HEIGHT, SYSTEM_PROMPT, ANTHROPIC_API_KEY, TASKS = _get_benchmark_imports()

        # 1. Load parent trace
        try:
            parent_trace = config.tree.load_trace(config.parent_trace_id)
        except Exception as e:
            return BranchExecutionResult(
                success=False,
                new_trace_id="",
                steps_count=0,
                error=f"Failed to load parent trace: {e}"
            )

        # Get steps to replay (before branch point)
        replay_steps = config.tree.get_steps_for_replay(
            config.parent_trace_id,
            config.branch_point_step
        )

        # Validate branch point
        parent_steps = parent_trace.get("steps", [])
        if config.branch_point_step < 1 or config.branch_point_step > len(parent_steps):
            return BranchExecutionResult(
                success=False,
                new_trace_id="",
                steps_count=0,
                error=f"Invalid branch point {config.branch_point_step}, trace has {len(parent_steps)} steps"
            )

        # 2. Determine app and start server
        app = ServerManager.detect_app_from_trace(parent_trace)
        self._report_progress(f"Starting {app} server...", 0, 0)

        try:
            if not self.server_manager.start_server(app, wait_ready=True, timeout=60):
                return BranchExecutionResult(
                    success=False,
                    new_trace_id="",
                    steps_count=0,
                    error=f"Failed to start {app} server (timeout)"
                )
        except Exception as e:
            return BranchExecutionResult(
                success=False,
                new_trace_id="",
                steps_count=0,
                error=f"Failed to start {app} server: {e}"
            )

        # 3. Create branch placeholder
        new_trace_id = config.tree.create_branch(
            parent_trace_id=config.parent_trace_id,
            branch_point_step=config.branch_point_step,
            intervention=config.intervention,
            label=config.label
        )

        # 4. Execute hybrid replay + live
        try:
            result = await self._run_hybrid_execution(
                config=config,
                new_trace_id=new_trace_id,
                parent_trace=parent_trace,
                replay_steps=replay_steps,
                app=app
            )
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return BranchExecutionResult(
                success=False,
                new_trace_id=new_trace_id,
                steps_count=0,
                error=str(e)
            )

    async def _run_hybrid_execution(
        self,
        config: BranchExecutionConfig,
        new_trace_id: str,
        parent_trace: dict,
        replay_steps: list[dict],
        app: str
    ) -> BranchExecutionResult:
        """Core execution: replay then live.

        This method:
        1. Launches browser
        2. Resets app state
        3. Replays steps deterministically
        4. Applies intervention
        5. Continues with live agent execution
        6. Saves the new trace
        """
        ComputerUseAgent, DISPLAY_WIDTH, DISPLAY_HEIGHT, SYSTEM_PROMPT, ANTHROPIC_API_KEY, TASKS = _get_benchmark_imports()

        meta = parent_trace.get("meta", {})
        api_url = self.server_manager.get_url(app)
        target_url = meta.get("target_url", api_url)
        discoverability = meta.get("discoverability", "navbar")
        capability = meta.get("capability", "advantage")

        # Determine model for live execution
        if config.intervention.type == InterventionType.MODEL_SWAP and config.intervention.model:
            model = config.intervention.model
        else:
            # Use parent's model
            model = "sonnet"  # Default
            parent_model = meta.get("model", "")
            if "haiku" in parent_model.lower():
                model = "haiku"

        # Get system prompt
        system_prompt = meta.get("system_prompt", SYSTEM_PROMPT)
        if config.intervention and config.intervention.type == InterventionType.GUIDANCE_PATCH:
            if config.intervention.system_prompt:
                system_prompt = config.intervention.system_prompt

        # Setup debug directory for new branch
        branch_info = config.tree._index.traces.get(new_trace_id, {})
        branch_dir = Path(branch_info.get("path", config.tree.base_dir / "branches" / new_trace_id))
        branch_dir.mkdir(parents=True, exist_ok=True)

        total_replay_steps = len(replay_steps)
        self._report_progress("Launching browser...", 0, total_replay_steps)

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=config.headless,
                args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"]
            )

            context = await browser.new_context(
                viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
                user_agent="CommerceACIBenchmark/Branch/1.0"
            )

            page = await context.new_page()

            # Create agent WITHOUT API key for replay phase
            agent = ComputerUseAgent(
                page,
                api_key=None,  # No LLM calls during replay
                debug_dir=branch_dir,
                system_prompt=system_prompt,
                model=model
            )

            # === PHASE 1: Reset and Navigate ===
            self._report_progress("Resetting app state...", 0, total_replay_steps)
            await agent.reset_session(api_url, discoverability=discoverability, capability=capability)
            await page.goto(target_url)
            await page.wait_for_load_state("networkidle")

            # === PHASE 2: Deterministic Replay ===
            for i, step in enumerate(replay_steps):
                self._report_progress(f"Replaying step {i + 1}/{total_replay_steps}...", i + 1, total_replay_steps)

                tool_uses = ((step.get("assistant") or {}).get("tool_uses") or [])
                for idx, tu in enumerate(tool_uses, start=1):
                    tool_input = tu.get("input") or {}
                    # Skip screenshot-only actions during replay
                    if tool_input.get("action") != "screenshot":
                        await agent._execute_computer_action(tool_input, action_index=idx)

                await asyncio.sleep(0.3)

            if config.rehydrate_conversation and replay_steps:
                trace_dir = self._get_trace_dir(config.tree, config.parent_trace_id)
                hydrate_trace = {
                    "meta": meta,
                    "steps": replay_steps,
                    "_trace_dir": trace_dir,
                }
                messages = trace_to_messages(hydrate_trace, len(replay_steps))
                agent.conversation_history = messages
                if messages:
                    agent.initial_prompt = messages[0]

            # === PHASE 3: Apply Intervention at Step N ===
            self._report_progress("Applying intervention...", total_replay_steps, total_replay_steps)

            # Now set up API key for live execution
            if not ANTHROPIC_API_KEY:
                await browser.close()
                return BranchExecutionResult(
                    success=False,
                    new_trace_id=new_trace_id,
                    steps_count=total_replay_steps,
                    error="ANTHROPIC_API_KEY not set"
                )

            agent.client = Anthropic(api_key=ANTHROPIC_API_KEY)

            # Prepare intervention parameters
            extra_text = None
            if config.intervention.type == InterventionType.PROMPT_INSERT:
                extra_text = config.intervention.prompt_text

            forced_action = None
            if config.intervention.type == InterventionType.TOOL_OVERRIDE:
                forced_action = config.intervention.forced_action

            if config.intervention.type == InterventionType.GUIDANCE_PATCH and config.intervention.system_prompt:
                agent.system_prompt = config.intervention.system_prompt

            # === PHASE 4: Live Execution from Step N ===
            instruction = meta.get("task_instruction", "Continue the task")
            live_steps_executed = 0
            success = False

            # Get task verifier if available
            task_id = meta.get("task_id")
            verifier = self._get_verifier_for_task(task_id, TASKS)

            total_live_steps = config.max_iterations
            for iteration in range(config.max_iterations):
                # Check if done via verifier
                if verifier:
                    state = await agent.get_ground_truth(api_url)
                    if state and verifier(state):
                        success = True
                        break

                # Report progress
                self._report_progress(
                    f"Live execution step {live_steps_executed + 1}...",
                    total_replay_steps + live_steps_executed + 1,
                    total_replay_steps + total_live_steps
                )

                # Handle intervention on first live step
                if iteration == 0 and forced_action:
                    # Execute forced action directly (skip model)
                    await agent._execute_computer_action(forced_action, action_index=1)
                    live_steps_executed += 1
                    # Don't apply extra_text after forced action
                    extra_text = None
                else:
                    # Normal agent step
                    action, is_done = await agent.run_step(
                        instruction,
                        extra_user_text=extra_text if iteration == 0 else None
                    )
                    live_steps_executed += 1

                    if is_done:
                        # Agent declared done - verify
                        if verifier:
                            state = await agent.get_ground_truth(api_url)
                            success = state and verifier(state)
                        else:
                            # No verifier, trust agent
                            success = True
                        break

                await asyncio.sleep(0.3)

            await browser.close()

        # === PHASE 5: Save Branch Trace ===
        total_steps = total_replay_steps + live_steps_executed

        # Build trace metadata
        trace_meta = {
            **meta,
            "trace_id": new_trace_id,
            "parent_trace_id": config.parent_trace_id,
            "branch_point_step": config.branch_point_step,
            "intervention": config.intervention.to_dict() if config.intervention else None,
            "is_branched": True,
            "branch_depth": self._get_branch_depth(config.tree, config.parent_trace_id) + 1,
            "created_at": datetime.now().isoformat(),
            "model": agent.model,
        }
        if config.intervention and config.intervention.type == InterventionType.GUIDANCE_PATCH:
            if config.intervention.system_prompt:
                trace_meta["system_prompt"] = config.intervention.system_prompt
            if config.intervention.guidance_fragments is not None:
                trace_meta["guidance_fragments"] = config.intervention.guidance_fragments
            if config.intervention.guidance_patch is not None:
                trace_meta["guidance_patch"] = config.intervention.guidance_patch

        trace_data = agent.export_trace(trace_meta)
        trace_data["schema_version"] = "trace.v2"
        trace_data["trace_version"] = "v2"
        trace_data["trace_id"] = new_trace_id
        trace_data["parent_trace_id"] = config.parent_trace_id
        trace_data["branch_point_step"] = config.branch_point_step
        trace_data["intervention"] = config.intervention.to_dict() if config.intervention else None

        config.tree.save_branch_trace(new_trace_id, trace_data, success=success)

        return BranchExecutionResult(
            success=success,
            new_trace_id=new_trace_id,
            steps_count=total_steps,
            trace_path=branch_dir / "trace.json"
        )

    def _get_verifier_for_task(self, task_id: str, tasks: list) -> Optional[Callable]:
        """Get verifier function for a task ID."""
        if not task_id:
            return None
        for t in tasks:
            if t.get("id") == task_id:
                return t.get("verifier")
        return None

    def _get_branch_depth(self, tree: TraceTree, trace_id: str) -> int:
        """Calculate branch depth by counting ancestors."""
        ancestors = tree.get_ancestors(trace_id)
        return len(ancestors)

    def _get_trace_dir(self, tree: TraceTree, trace_id: str) -> Optional[Path]:
        """Resolve the directory containing a trace's artifacts."""
        info = tree._index.traces.get(trace_id, {})
        trace_path = info.get("path")
        return Path(trace_path) if trace_path else None


def run_branch_sync(config: BranchExecutionConfig, progress_callback=None) -> BranchExecutionResult:
    """Synchronous wrapper for branch execution.

    Useful for calling from Streamlit which runs in a sync context.
    """
    from server_manager import get_server_manager

    executor = BranchExecutor(
        server_manager=get_server_manager(),
        progress_callback=progress_callback
    )

    # Run in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, executor.execute_branch(config))
                return future.result()
        else:
            return loop.run_until_complete(executor.execute_branch(config))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(executor.execute_branch(config))
