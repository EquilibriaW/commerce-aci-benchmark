#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commerce ACI Benchmark - Claude Computer Use Edition

Uses Claude Sonnet 4.5 with computer use (vision + coordinates) to test
agent performance on e-commerce tasks.

Requirements:
- ANTHROPIC_API_KEY environment variable
- BENCHMARK_SECRET environment variable (default: sk-bench-123)
- Treatment server running on localhost:3000
- Baseline server running on localhost:3001
"""

import asyncio
import base64
import os
import sys
import argparse
import json
import random
import statistics
import time
import uuid
import hashlib
import tomllib
from datetime import datetime

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import Optional, Any

from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError
from playwright.async_api import async_playwright, Page, Browser
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

# --- CONFIGURATION ---
URL_TREATMENT = os.getenv("URL_TREATMENT", "http://localhost:3000")
URL_TREATMENT_DOCS = os.getenv("URL_TREATMENT_DOCS", "http://localhost:3002")
URL_BASELINE = os.getenv("URL_BASELINE", "http://localhost:3001")

BENCHMARK_SECRET = os.getenv("BENCHMARK_SECRET", "sk-bench-123")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Computer use display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800

MAX_ITERATIONS = 18
RUNS_PER_TASK = 1

# Retry configuration for API calls
MAX_API_RETRIES = 5          # Maximum retry attempts before failing
INITIAL_BACKOFF_S = 1.0      # Initial backoff in seconds
MAX_BACKOFF_S = 60.0         # Maximum backoff in seconds
BACKOFF_MULTIPLIER = 2.0     # Exponential backoff multiplier
JITTER_FACTOR = 0.25         # Random jitter factor (0-1)

# Debug settings
DEBUG_SCREENSHOTS = True
DEBUG_DIR = Path("debug_screenshots")

console = Console()

# System prompt (cached across steps)
SYSTEM_PROMPT = """You are an AI agent operating a computer to complete e-commerce tasks.
Your core objectives are TRUST, SPEED, and CORRECTNESS.

You have access to a web browser and can interact using the computer tool.

GUIDELINES:
1. **Analyze the Interface**: Scan the screen. You may encounter standard graphical interfaces or specialized high-density agent interfaces. Use whichever interface allows you to complete the task most efficiently.
2. **Prioritize Speed**: Choose the path that requires the fewest steps (clicks, scrolls, navigations) to achieve the goal.
3. **Ensure Correctness**: Verify that the product, variant (size/color), and quantity match the request exactly before finalizing.
4. **Execution**: Use the computer tool to click coordinates, type text, or scroll.

When the task is successfully completed, respond with "TASK_COMPLETE"."""

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-3-5-haiku-20241022"
}
DEFAULT_MODEL = MODEL_ALIASES["sonnet"]

ASK_USER_TAG = "ASK_USER:"


def _extract_user_request(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(ASK_USER_TAG):
            question = stripped[len(ASK_USER_TAG):].strip()
            return question or None
    return None

# --- VERIFIABLE TASK SUITE ---
# Benchmark products:
# - black-t-shirt: $20.00, sizes S/M/L
# - acme-cup: $15.00, no variants
# - hoodie: $50.00, no variants
#
# NOTE: Verifiers check `last_order` (completed checkout) not just `cart` (items added).
# This ensures agents must complete the full checkout flow with name/email,
# making the comparison between human UI and agent UI fair.

def get_order_items(state: dict) -> list:
    """Get items from completed order, or empty list if no order."""
    order = state.get('last_order')
    if order and order.get('items'):
        return order['items']
    return []

def get_order_total(state: dict) -> int:
    """Get total from completed order, or 0 if no order."""
    order = state.get('last_order')
    if order:
        return order.get('total_price_cents', 0)
    return 0

def has_completed_order(state: dict) -> bool:
    """Check if checkout was completed (order exists with customer info)."""
    order = state.get('last_order')
    return order is not None and order.get('customer') is not None

TASKS = [
    {
        "id": "t01_find_add_simple",
        "instruction": "Buy me a black T-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i['quantity'] >= 1
                for i in get_order_items(s))
        )
    },
    {
        "id": "t02_variant_size_l",
        "instruction": "Buy me a large black T-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i.get('variant') == 'L'
                for i in get_order_items(s))
        )
    },
    {
        "id": "t03_cart_total_check",
        "instruction": "Get me two acme cups and a hoodie, for less than 90 dollars",
        "verifier": lambda s: (
            has_completed_order(s) and
            get_order_total(s) == 8000
        )
    },
    {
        "id": "t04_quantity_adjust",
        "instruction": "Add 2 acme cups to cart, then change the quantity to 1, and checkout",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'acme-cup' and i['quantity'] == 1
                for i in get_order_items(s)) and
            get_order_total(s) == 1500  # 1 cup at $15
        )
    },
    {
        "id": "t05_remove_then_buy_variant",
        "instruction": "Add a hoodie to cart, remove it, then buy a medium black T-shirt instead",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i.get('variant') == 'M'
                for i in get_order_items(s)) and
            not any(i['slug'] == 'hoodie' for i in get_order_items(s)) and
            get_order_total(s) == 2000  # 1 M t-shirt at $20
        )
    },
    {
        "id": "t06_overbuy_then_fix",
        "instruction": "Add 2 hoodies to cart. Actually, I only want 1 hoodie and 2 cups. Fix my cart and checkout.",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'hoodie' and i['quantity'] == 1
                for i in get_order_items(s)) and
            any(i['slug'] == 'acme-cup' and i['quantity'] == 2
                for i in get_order_items(s)) and
            get_order_total(s) == 8000  # 1 hoodie ($50) + 2 cups ($30)
        )
    }
]


class ComputerUseAgent:
    """Agent that uses Claude's computer use capability with screenshots."""

    # UI actions that count for metrics (exclude screenshot/wait/mouse_move)
    UI_ACTIONS = {'left_click', 'right_click', 'double_click', 'triple_click',
                  'left_click_drag', 'type', 'key', 'scroll'}
    # Max conversation cycles to keep (preserve initial prompt, truncate older cycles)
    MAX_HISTORY_CYCLES = 5

    # Supported models
    SUPPORTED_MODELS = {
        "sonnet": "claude-sonnet-4-5-20250929",
        "haiku": "claude-3-5-haiku-20241022",
    }
    DEFAULT_MODEL = "sonnet"

    def __init__(self, page: Page, api_key: Optional[str] = None, debug_dir: Optional[Path] = None,
                 system_prompt: str = SYSTEM_PROMPT, model: str = DEFAULT_MODEL):
        """Create a computer-use agent.

        Args:
            page: Playwright page to control.
            api_key: Anthropic API key. If None/empty, the agent can still be used
                for deterministic *replay execution* (re-running recorded UI actions)
                but cannot call the LLM.
            debug_dir: Optional directory for saving debug artifacts.
            system_prompt: System prompt for the agent.
            model: Model to use - "sonnet" (claude-sonnet-4-5) or "haiku" (claude-haiku-3-5).
        """
        self.page = page
        self.client = Anthropic(api_key=api_key) if api_key else None
        self.conversation_history = []
        self.initial_prompt = None  # Store the initial user prompt (preserved across truncation)
        self.entered_agent_view = False
        self.first_agent_view_step = None  # Step when agent UI was first visited (adoption timing)
        self.agent_action_requests = 0
        self.step_count = 0
        self.model_calls = 0  # Number of LLM API calls
        self.ui_actions = 0   # Number of real UI actions (clicks/drag/type/key/scroll)
        self.debug_dir = debug_dir
        self.action_log = []
        self.system_prompt = system_prompt
        # Resolve model name to full model ID
        self.model = self.SUPPORTED_MODELS.get(model, model)

        # Structured trace ("VCR for agents"). Written to trace.json at end of run.
        # This is intentionally lightweight: screenshots are stored as files and
        # referenced by path to avoid huge JSON blobs.
        # Schema v2 supports branching fields (parent_trace_id, intervention, etc.)
        self.trace: dict[str, Any] = {
            "schema_version": "trace.v2",
            "trace_version": "v2",
            "trace_id": str(uuid.uuid4()),
            "parent_trace_id": None,  # Set when creating branches
            "branch_point_step": None,  # Step number where branch diverged
            "intervention": None,  # Intervention applied at branch point
            "meta": {},
            "steps": []
        }

        # Track network requests to /agent/actions
        self.page.on("request", self._handle_request)

    def _handle_request(self, request):
        if request.method == "POST" and "/agent/actions/" in request.url:
            self.agent_action_requests += 1

    async def take_screenshot(self) -> tuple[str, bytes]:
        """Take a screenshot and return as (base64, raw_bytes)."""
        screenshot_bytes = await self.page.screenshot(type="png")
        screenshot_b64 = base64.standard_b64encode(screenshot_bytes).decode("utf-8")
        return screenshot_b64, screenshot_bytes

    def _save_debug_screenshot(self, screenshot_bytes: bytes, step: int, suffix: str = "") -> Optional[str]:
        """Save screenshot to debug directory.

        Returns the saved path as a string, or None if debug_dir is not set.
        """
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = self.debug_dir / f"step_{step:02d}{suffix}.png"
            screenshot_path.write_bytes(screenshot_bytes)
            return str(screenshot_path)
        return None

    def _log_action(self, step: int, action: str, details: str):
        """Log an action for debugging."""
        log_entry = f"Step {step:02d}: {action} - {details}"
        self.action_log.append(log_entry)
        console.print(f"    [dim]{log_entry}[/dim]")

    async def reset_session(
        self,
        base_url: str,
        discoverability: str = "navbar",
        capability: str = "advantage",
        variant_seed: Optional[int] = None,
        variant_level: Optional[int] = None
    ):
        """Reset the session via API using Playwright's request context for cookie consistency.

        Args:
            base_url: Base URL of the app
            discoverability: "navbar" (visible link) or "hidden" (no link)
            capability: "advantage" (agent actions enabled) or "parity" (read-only)
            variant_seed: Optional seed for deterministic UI variants
            variant_level: Optional difficulty level for UI variants
        """
        # Use Playwright's request context - cookies are automatically shared with browser
        headers = {
            "X-Benchmark-Secret": BENCHMARK_SECRET,
            "X-Benchmark-Discoverability": discoverability,
            "X-Benchmark-Capability": capability
        }
        if variant_seed is not None:
            headers["X-Benchmark-Variant-Seed"] = str(variant_seed)
        if variant_level is not None:
            headers["X-Benchmark-Variant-Level"] = str(variant_level)

        resp = await self.page.context.request.post(
            f"{base_url}/agent/reset",
            headers=headers
        )
        if resp.status != 200:
            raise Exception(f"Failed to reset session: {resp.status} {await resp.text()}")

    async def get_ground_truth(self, base_url: str) -> dict:
        """Get the current cart state from the API using Playwright's request context."""
        # Use Playwright's request context - cookies are automatically shared
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = await self.page.context.request.get(
                    f"{base_url}/agent/state",
                    headers={"X-Benchmark-Secret": BENCHMARK_SECRET}
                )
                if resp.status == 200:
                    return await resp.json()
                return {}
            except Exception as e:
                last_error = e
                await asyncio.sleep(0.5 * (2 ** attempt))
        return {"error": str(last_error)} if last_error else {}

    def _truncate_history(self):
        """Truncate conversation history while preserving initial prompt.

        Keeps the initial user message (with task instruction) and removes
        complete interaction cycles from the front. A cycle is defined as:
        - An assistant message (possibly containing tool_use)
        - Followed by a user message (possibly containing tool_result)

        Never breaks tool_use/tool_result pairings. Scans from the end to
        identify complete cycles rather than assuming fixed message lengths.
        """
        if len(self.conversation_history) <= 1:
            return

        messages_after_initial = self.conversation_history[1:]

        # We want to keep at most MAX_HISTORY_CYCLES complete cycles
        # Each cycle is typically assistant + user (2 messages)
        max_messages = self.MAX_HISTORY_CYCLES * 2

        if len(messages_after_initial) <= max_messages:
            return

        # Scan backwards to find complete cycles
        # A complete cycle ends with: assistant message (with or without tool_use)
        # followed by user message (with or without tool_result)
        remaining = list(messages_after_initial)

        while len(remaining) > max_messages:
            # Check if we can safely remove the first message(s)
            if len(remaining) < 2:
                break

            first = remaining[0]
            second = remaining[1] if len(remaining) > 1 else None

            # Check if first message is assistant with tool_use
            first_has_tool_use = False
            if first.get("role") == "assistant":
                content = first.get("content", [])
                first_has_tool_use = any(
                    isinstance(c, dict) and c.get("type") == "tool_use"
                    for c in (content if isinstance(content, list) else [])
                )

            # Check if second message is user with tool_result
            second_has_tool_result = False
            if second and second.get("role") == "user":
                content = second.get("content", [])
                second_has_tool_result = any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in (content if isinstance(content, list) else [])
                )

            # If assistant has tool_use, we must remove both messages together
            if first_has_tool_use:
                if second_has_tool_result:
                    # Safe to remove the pair
                    remaining = remaining[2:]
                else:
                    # tool_use without matching tool_result - shouldn't happen,
                    # but if it does, just remove the assistant message
                    remaining = remaining[1:]
            elif first.get("role") == "assistant":
                # Assistant without tool_use - safe to remove just this one
                remaining = remaining[1:]
            elif first.get("role") == "user":
                # User message at the start
                first_content = first.get("content", [])
                has_tool_result = any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in (first_content if isinstance(first_content, list) else [])
                )
                if has_tool_result:
                    # Orphaned tool_result - remove it
                    remaining = remaining[1:]
                else:
                    # Regular user message - safe to remove
                    remaining = remaining[1:]
            else:
                # Unknown role - just remove it
                remaining = remaining[1:]

        self.conversation_history = [self.conversation_history[0]] + remaining

    async def run_step(self, instruction: str, extra_user_text: Optional[str] = None) -> tuple[str, bool]:
        """
        Run one step of the agent loop.
        Returns (action_taken, is_done).
        """
        if self.client is None:
            raise RuntimeError("ComputerUseAgent has no Anthropic client. Provide ANTHROPIC_API_KEY for LLM execution.")

        self.step_count += 1

        # Check if we're in agent view (track first adoption step)
        if "/agent" in self.page.url:
            if not self.entered_agent_view:
                self.first_agent_view_step = self.step_count
            self.entered_agent_view = True

        # Check if last message was a tool_result (which already includes screenshot)
        # In that case, don't add another user message - just call the API
        last_msg = self.conversation_history[-1] if self.conversation_history else None
        last_was_tool_result = False
        if last_msg and last_msg.get("role") == "user":
            content = last_msg.get("content", [])
            if isinstance(content, list) and content:
                last_was_tool_result = any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in content
                )

        # Always take and save a debug screenshot at the start of each step
        screenshot_b64, screenshot_bytes = await self.take_screenshot()
        pre_path = self._save_debug_screenshot(screenshot_bytes, self.step_count, suffix="_pre")

        # The screenshot actually visible to the model for this step depends on the
        # message structure. If the last message was a tool_result (which already includes
        # a screenshot), we do NOT attach a new screenshot message; in that case the model
        # sees the previous step's last tool_result screenshot. For replay/counterfactual
        # correctness, record that as the observation screenshot.
        prompt_obs_path = pre_path
        if last_was_tool_result and self.trace["steps"]:
            prev = self.trace["steps"][-1]
            prev_tool_results = prev.get("tool_results") or []
            last_img = (prev_tool_results[-1].get("screenshot_path") if prev_tool_results else None)
            if last_img:
                prompt_obs_path = last_img

        step_trace: dict[str, Any] = {
            "step": self.step_count,
            "page_url": self.page.url,
            "timestamp": datetime.now().isoformat(),
            "user_extra": extra_user_text,
            "prompt_hash": hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
            "observation": {
                "screenshot_path": prompt_obs_path,
                "debug_screenshot_path": pre_path,
                "display_width": DISPLAY_WIDTH,
                "display_height": DISPLAY_HEIGHT,
            },
            "assistant": {
                "text": None,
                "tool_uses": []
            },
            "tool_results": []
        }

        if last_was_tool_result and extra_user_text:
            # If the last user message was already a tool_result (includes screenshot), we still
            # want to inject a user utterance (e.g., adversarial fuzzing) as a separate message.
            self.conversation_history.append({
                "role": "user",
                "content": [{"type": "text", "text": extra_user_text}]
            })
            self._truncate_history()

        if not last_was_tool_result:
            # Build messages - use system parameter for system prompt
            if not self.conversation_history:
                # First message - task instruction + screenshot
                extra = f"\n\nUSER UPDATE: {extra_user_text}" if extra_user_text else ""
                self.initial_prompt = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"TASK: {instruction}{extra}\n\nHere is the current state of the webpage:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64
                            }
                        }
                    ]
                }
                self.conversation_history = [self.initial_prompt]
            else:
                # Follow-up message with new screenshot (only if last wasn't tool_result)
                prefix = f"USER UPDATE: {extra_user_text}\n\n" if extra_user_text else ""
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prefix + "Here is the current state of the page after your last action. Continue with the task or say TASK_COMPLETE if done."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64
                            }
                        }
                    ]
                })
                # Truncate history to prevent context overflow
                self._truncate_history()

        # Call Claude with computer use - use system= parameter
        # Bounded retry with exponential backoff + jitter
        response = None
        last_error = None
        backoff = INITIAL_BACKOFF_S

        for attempt in range(MAX_API_RETRIES):
            try:
                self.model_calls += 1
                # Run sync API call in a thread to avoid blocking the event loop
                # Use prompt caching for the system prompt (cache_control with ephemeral type)
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=4096,
                    system=[
                        {
                            "type": "text",
                            "text": self.system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    extra_headers={"anthropic-beta": "computer-use-2025-01-24,prompt-caching-2024-07-31"},
                    tools=[
                        {
                            "type": "computer_20250124",
                            "name": "computer",
                            "display_width_px": DISPLAY_WIDTH,
                            "display_height_px": DISPLAY_HEIGHT,
                            "display_number": 1
                        }
                    ],
                    messages=self.conversation_history
                )
                break  # Success - exit retry loop

            except RateLimitError as e:
                # Transient - retry with backoff
                last_error = e
                jitter = backoff * JITTER_FACTOR * random.random()
                wait_time = min(backoff + jitter, MAX_BACKOFF_S)
                console.print(f"    [yellow]Rate limited (attempt {attempt + 1}/{MAX_API_RETRIES}), waiting {wait_time:.1f}s...[/yellow]")
                await asyncio.sleep(wait_time)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_S)

            except APIConnectionError as e:
                # Network error - retry with backoff
                last_error = e
                jitter = backoff * JITTER_FACTOR * random.random()
                wait_time = min(backoff + jitter, MAX_BACKOFF_S)
                console.print(f"    [yellow]Connection error (attempt {attempt + 1}/{MAX_API_RETRIES}), waiting {wait_time:.1f}s...[/yellow]")
                await asyncio.sleep(wait_time)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_S)

            except APIError as e:
                last_error = e
                status = getattr(e, 'status_code', None)
                if status and 500 <= status < 600:
                    # Server error - retry with backoff
                    jitter = backoff * JITTER_FACTOR * random.random()
                    wait_time = min(backoff + jitter, MAX_BACKOFF_S)
                    console.print(f"    [yellow]API {status} error (attempt {attempt + 1}/{MAX_API_RETRIES}), waiting {wait_time:.1f}s...[/yellow]")
                    await asyncio.sleep(wait_time)
                    backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_S)
                else:
                    # Client error (4xx except rate limit) - fail fast
                    console.print(f"    [red]API error {status}: {e}[/red]")
                    raise

            except Exception as e:
                # Unknown error - fail fast with actionable log
                console.print(f"    [red]Unexpected error: {type(e).__name__}: {e}[/red]")
                raise

        if response is None:
            # Exhausted all retries
            console.print(f"    [red]Failed after {MAX_API_RETRIES} retries. Last error: {last_error}[/red]")
            raise last_error or Exception("API call failed after max retries")

        # Process response - handle multiple tool_use blocks
        assistant_content = []
        tool_uses = []
        action_taken = "none"
        is_done = False

        assistant_text_chunks: list[str] = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                assistant_text_chunks.append(block.text)
                if "TASK_COMPLETE" in block.text.upper():
                    is_done = True
                    action_taken = "TASK_COMPLETE"
                    self._log_action(self.step_count, "TASK_COMPLETE", "Agent declared task complete")

            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
                tool_uses.append(block)

        step_trace["assistant"]["text"] = "\n".join(assistant_text_chunks) if assistant_text_chunks else None
        user_request = _extract_user_request(step_trace["assistant"]["text"])
        if user_request:
            step_trace["assistant_user_request"] = {"question": user_request, "tag": ASK_USER_TAG}
            if action_taken == "none":
                action_taken = "ask_user"

        # Add assistant message first (contains all tool_use blocks)
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })

        # Execute all tool_uses sequentially and collect results
        if tool_uses:
            tool_results = []
            for action_idx, block in enumerate(tool_uses, start=1):
                # Log the action
                action_type = block.input.get('action', 'unknown')
                coord = block.input.get('coordinate', [0, 0])
                text = block.input.get('text', '')

                # Count UI actions
                if action_type in self.UI_ACTIONS:
                    self.ui_actions += 1

                if action_type in ['left_click', 'right_click', 'double_click']:
                    self._log_action(self.step_count, action_type, f"at ({coord[0]}, {coord[1]})")
                elif action_type == 'type':
                    self._log_action(self.step_count, action_type, f"'{text[:30]}...'")
                elif action_type == 'scroll':
                    scroll_dir = block.input.get('scroll_direction', 'down')
                    scroll_amt = block.input.get('scroll_amount', 0)
                    self._log_action(self.step_count, action_type, f"{scroll_dir} by {scroll_amt}")
                elif action_type == 'key':
                    key = block.input.get('key', '')
                    self._log_action(self.step_count, action_type, f"'{key}'")
                else:
                    self._log_action(self.step_count, action_type, str(block.input)[:50])

                step_trace["assistant"]["tool_uses"].append({
                    "tool_use_id": block.id,
                    "name": block.name,
                    "input": block.input
                })

                # Execute the computer action
                tool_result, post_path, result_text = await self._execute_computer_action(block.input, action_index=action_idx)
                action_taken = action_type

                step_trace["tool_results"].append({
                    "tool_use_id": block.id,
                    "result_text": result_text,
                    "screenshot_path": post_path
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result
                })

            # Add all tool results in a single user message
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

        # Persist trace step
        self.trace["steps"].append(step_trace)

        return action_taken, is_done

    async def _execute_computer_action(self, action_input: dict, action_index: int = 1) -> tuple[list, Optional[str], str]:
        """Execute a computer use action and return result with screenshot.

        Returns a list of content blocks including text result and new screenshot.
        This is CRITICAL - Claude needs to see the result of its action.
        """
        action = action_input.get("action")
        result_text = ""

        try:
            if action == "screenshot":
                result_text = "Screenshot taken successfully."

            elif action == "mouse_move":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.move(x, y)
                result_text = f"Moved mouse to ({x}, {y})"

            elif action == "left_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.click(x, y)
                await asyncio.sleep(2.0)  # Wait for any navigation/updates
                result_text = f"Left clicked at ({x}, {y})"

            elif action == "left_click_drag":
                start = action_input.get("start_coordinate", [0, 0])
                end = action_input.get("coordinate", [0, 0])
                await self.page.mouse.move(start[0], start[1])
                await self.page.mouse.down()
                await self.page.mouse.move(end[0], end[1])
                await self.page.mouse.up()
                result_text = f"Dragged from {start} to {end}"

            elif action == "right_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.click(x, y, button="right")
                result_text = f"Right clicked at ({x}, {y})"

            elif action == "double_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.dblclick(x, y)
                result_text = f"Double clicked at ({x}, {y})"

            elif action == "triple_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.click(x, y, click_count=3)
                result_text = f"Triple clicked at ({x}, {y})"

            elif action == "scroll":
                x = action_input.get("coordinate", [DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2])[0]
                y = action_input.get("coordinate", [DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2])[1]
                # New API uses scroll_direction and scroll_amount
                scroll_direction = action_input.get("scroll_direction", "down")
                scroll_amount = action_input.get("scroll_amount", 3)
                # Convert direction+amount to delta pixels (100px per unit)
                delta_x, delta_y = 0, 0
                pixels_per_unit = 100
                if scroll_direction == "down":
                    delta_y = scroll_amount * pixels_per_unit
                elif scroll_direction == "up":
                    delta_y = -scroll_amount * pixels_per_unit
                elif scroll_direction == "right":
                    delta_x = scroll_amount * pixels_per_unit
                elif scroll_direction == "left":
                    delta_x = -scroll_amount * pixels_per_unit
                await self.page.mouse.move(x, y)
                await self.page.mouse.wheel(delta_x, delta_y)
                await asyncio.sleep(0.5)  # Wait for scroll to complete
                result_text = f"Scrolled {scroll_direction} by {scroll_amount} at ({x}, {y})"

            elif action == "type":
                text = action_input.get("text", "")
                # Use delay between keystrokes (12ms like Anthropic reference)
                await self.page.keyboard.type(text, delay=12)
                result_text = f"Typed: {text[:50]}..."

            elif action == "key":
                key = action_input.get("key", "")
                # Map common key names
                key_map = {
                    "Return": "Enter",
                    "BackSpace": "Backspace",
                    "space": " ",
                }
                key = key_map.get(key, key)
                if key:
                    await self.page.keyboard.press(key)
                    result_text = f"Pressed key: {key}"
                else:
                    result_text = "Warning: empty key ignored"

            elif action == "hold_key":
                key = action_input.get("key", "")
                await self.page.keyboard.down(key)
                result_text = f"Holding key: {key}"

            elif action == "release_key":
                key = action_input.get("key", "")
                await self.page.keyboard.up(key)
                result_text = f"Released key: {key}"

            elif action == "wait":
                duration = action_input.get("duration", 1000)  # milliseconds
                await asyncio.sleep(duration / 1000)
                result_text = f"Waited {duration}ms"

            else:
                result_text = f"Unknown action: {action}"

        except Exception as e:
            result_text = f"Action failed: {str(e)}"

        # CRITICAL: Take new screenshot after action so Claude sees the result
        screenshot_b64, screenshot_bytes = await self.take_screenshot()

        # Save post-action screenshot for debugging (unique per action)
        post_path = self._save_debug_screenshot(screenshot_bytes, self.step_count, suffix=f"_action_{action_index:02d}")

        content = [
            {"type": "text", "text": result_text},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64
                }
            }
        ]

        return content, post_path, result_text

    def export_trace(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Return a structured trace payload suitable for saving to JSON.

        The trace uses schema v2 which includes:
        - trace_id: Unique identifier for this trace
        - parent_trace_id: ID of parent trace (for branches)
        - branch_point_step: Step where branch diverged (for branches)
        - intervention: Intervention applied at branch point (for branches)
        """
        # Include trace_id in meta for easier access
        meta["trace_id"] = self.trace["trace_id"]
        self.trace["meta"] = meta
        return self.trace


def _load_gate_policy(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data.get("policy") or data)


async def run_trial(
    task: dict,
    target_url: str,
    api_url: str,
    condition_name: str,
    run_num: int,
    discoverability: str = "navbar",
    capability: str = "advantage",
    variant_seed: Optional[int] = None,
    variant_level: Optional[int] = None,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = ComputerUseAgent.DEFAULT_MODEL,
    guidance_meta: Optional[dict[str, Any]] = None,
) -> dict:
    """Run a single trial of a task.

    Args:
        task: Task definition with id, instruction, verifier
        target_url: Starting URL for the agent
        api_url: Base URL for API calls (reset, state)
        condition_name: Human-readable condition name
        run_num: Run number (1-indexed)
        discoverability: "navbar" or "hidden" - controls agent UI visibility
        capability: "advantage" or "parity" - controls agent actions
        variant_seed: Seed for deterministic UI variants (optional)
        variant_level: Difficulty level for UI variants (optional)
        system_prompt: System prompt for the agent
        model: Model to use - "sonnet" or "haiku"
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Track if we're starting on /agent (forced adoption - show N/A for adoption metric)
    forced_agent_start = "/agent" in target_url

    # Set up debug directory with timestamp
    debug_dir = None
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if DEBUG_SCREENSHOTS:
        # Sanitize condition name for directory
        safe_condition = condition_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        debug_dir = DEBUG_DIR / safe_condition / task["id"] / f"run_{run_num:02d}_{run_timestamp}"

    # Start wall time tracking
    start_time = time.perf_counter()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"]
        )

        context = await browser.new_context(
            viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
            user_agent="CommerceACIBenchmark/1.0"
        )

        page = await context.new_page()
        agent = ComputerUseAgent(page, ANTHROPIC_API_KEY, debug_dir=debug_dir, system_prompt=system_prompt, model=model)

        # Reset session with experimental factors
        await agent.reset_session(
            api_url,
            discoverability=discoverability,
            capability=capability,
            variant_seed=variant_seed,
            variant_level=variant_level
        )

        # Navigate to starting page
        await page.goto(target_url)
        await page.wait_for_load_state("networkidle")

        steps = 0
        success = False

        for _ in range(MAX_ITERATIONS):
            # Check if task is complete
            state = await agent.get_ground_truth(api_url)
            if state and task["verifier"](state):
                success = True
                break

            # Run one agent step
            action, is_done = await agent.run_step(task["instruction"])
            steps += 1

            if is_done:
                # Final check
                state = await agent.get_ground_truth(api_url)
                if state and task["verifier"](state):
                    success = True
                break

            # Small delay between actions
            await asyncio.sleep(0.3)

        final_state = await agent.get_ground_truth(api_url)

        # End wall time tracking
        wall_time_seconds = time.perf_counter() - start_time

        # Save action log
        if debug_dir:
            log_path = debug_dir / "actions.log"
            log_path.write_text("\n".join(agent.action_log))

            # Save structured trace for deterministic replay + counterfactual analysis
            trace_path = debug_dir / "trace.json"
            trace_meta = {
                "run_timestamp": run_timestamp,
                "condition_name": condition_name,
                "task_id": task.get("id"),
                "task_instruction": task.get("instruction"),
                "target_url": target_url,
                "api_url": api_url,
                "discoverability": discoverability,
                "capability": capability,
                "model": agent.model,
                "anthropic_beta": "computer-use-2025-01-24,prompt-caching-2024-07-31",
                "display_width": DISPLAY_WIDTH,
                "display_height": DISPLAY_HEIGHT,
                "max_iterations": MAX_ITERATIONS,
                "system_prompt": agent.system_prompt,
                "variant_seed": variant_seed,
                "variant_level": variant_level,
                "success": success,
                "final_state": final_state,
            }
            if guidance_meta:
                trace_meta.update(guidance_meta)
            trace_path.write_text(json.dumps(agent.export_trace(trace_meta), indent=2))

        await browser.close()

        return {
            "success": success,
            "steps": steps,
            "entered_agent_view": agent.entered_agent_view,
            "first_agent_view_step": agent.first_agent_view_step,
            "agent_actions": agent.agent_action_requests,
            "model_calls": agent.model_calls,
            "ui_actions": agent.ui_actions,
            "wall_time_seconds": round(wall_time_seconds, 2),
            "forced_agent_start": forced_agent_start,
            "action_log": agent.action_log,
            "trace_path": str((debug_dir / "trace.json")) if debug_dir else None,
            "final_state": final_state
        }


async def main(args: argparse.Namespace):
    """Run the full benchmark."""
    if not ANTHROPIC_API_KEY:
        console.print("[red]ERROR: ANTHROPIC_API_KEY environment variable is required[/red]")
        return

    results = []

    # === FACTORIZED EXPERIMENTAL CONDITIONS ===
    # Three independent factors:
    #   app: baseline | treatment | treatment-docs
    #   discoverability: navbar | hidden (treatment/treatment-docs only)
    #   capability: advantage | parity (treatment/treatment-docs only)
    #
    # Plus entry point variations for adoption measurement:
    #   start: root | /agent (forced)

    CONDITIONS = [
        # --- Baseline (no factors - control condition) ---
        {
            "name": "Baseline/Root",
            "app": "baseline",
            "target_url": URL_BASELINE,
            "api_url": URL_BASELINE,
            "discoverability": "navbar",  # no-op for baseline
            "capability": "advantage",     # no-op for baseline
        },

        # --- Treatment (Terminal UI) - Full factorial ---
        # discoverability=navbar, capability=advantage (full features)
        {
            "name": "Treatment/Root/navbar/advantage",
            "app": "treatment",
            "target_url": URL_TREATMENT,
            "api_url": URL_TREATMENT,
            "discoverability": "navbar",
            "capability": "advantage",
        },
        {
            "name": "Treatment/Agent/navbar/advantage",
            "app": "treatment",
            "target_url": f"{URL_TREATMENT}/agent",
            "api_url": URL_TREATMENT,
            "discoverability": "navbar",
            "capability": "advantage",
        },
        # discoverability=hidden, capability=advantage
        {
            "name": "Treatment/Root/hidden/advantage",
            "app": "treatment",
            "target_url": URL_TREATMENT,
            "api_url": URL_TREATMENT,
            "discoverability": "hidden",
            "capability": "advantage",
        },
        # discoverability=navbar, capability=parity (read-only agent UI)
        {
            "name": "Treatment/Root/navbar/parity",
            "app": "treatment",
            "target_url": URL_TREATMENT,
            "api_url": URL_TREATMENT,
            "discoverability": "navbar",
            "capability": "parity",
        },
        {
            "name": "Treatment/Agent/navbar/parity",
            "app": "treatment",
            "target_url": f"{URL_TREATMENT}/agent",
            "api_url": URL_TREATMENT,
            "discoverability": "navbar",
            "capability": "parity",
        },
        # discoverability=hidden, capability=parity (worst case - no link, read-only)
        {
            "name": "Treatment/Root/hidden/parity",
            "app": "treatment",
            "target_url": URL_TREATMENT,
            "api_url": URL_TREATMENT,
            "discoverability": "hidden",
            "capability": "parity",
        },

        # --- Treatment-Docs (Documentation UI) - Full factorial ---
        # discoverability=navbar, capability=advantage (full features)
        {
            "name": "TreatmentDocs/Root/navbar/advantage",
            "app": "treatment-docs",
            "target_url": URL_TREATMENT_DOCS,
            "api_url": URL_TREATMENT_DOCS,
            "discoverability": "navbar",
            "capability": "advantage",
        },
        {
            "name": "TreatmentDocs/Agent/navbar/advantage",
            "app": "treatment-docs",
            "target_url": f"{URL_TREATMENT_DOCS}/agent",
            "api_url": URL_TREATMENT_DOCS,
            "discoverability": "navbar",
            "capability": "advantage",
        },
        # discoverability=hidden, capability=advantage
        {
            "name": "TreatmentDocs/Root/hidden/advantage",
            "app": "treatment-docs",
            "target_url": URL_TREATMENT_DOCS,
            "api_url": URL_TREATMENT_DOCS,
            "discoverability": "hidden",
            "capability": "advantage",
        },
        # discoverability=navbar, capability=parity
        {
            "name": "TreatmentDocs/Root/navbar/parity",
            "app": "treatment-docs",
            "target_url": URL_TREATMENT_DOCS,
            "api_url": URL_TREATMENT_DOCS,
            "discoverability": "navbar",
            "capability": "parity",
        },
        {
            "name": "TreatmentDocs/Agent/navbar/parity",
            "app": "treatment-docs",
            "target_url": f"{URL_TREATMENT_DOCS}/agent",
            "api_url": URL_TREATMENT_DOCS,
            "discoverability": "navbar",
            "capability": "parity",
        },
        # discoverability=hidden, capability=parity
        {
            "name": "TreatmentDocs/Root/hidden/parity",
            "app": "treatment-docs",
            "target_url": URL_TREATMENT_DOCS,
            "api_url": URL_TREATMENT_DOCS,
            "discoverability": "hidden",
            "capability": "parity",
        },
    ]


    # --- CLI filtering / configuration ---
    base_prompt = SYSTEM_PROMPT
    if getattr(args, "system_prompt_file", None):
        try:
            base_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]ERROR: Failed to read --system-prompt-file: {e}[/red]")
            return

    guidance_packs_raw = getattr(args, "guidance_packs", "")
    guidance_packs = [p.strip() for p in str(guidance_packs_raw).split(",") if p.strip()]
    guidance_fragments: list[str] = []
    guidance_meta = None
    if guidance_packs:
        try:
            from pack_api.loader import PackRegistry
            from pack_api.runtime import assemble_system_prompt, load_guidance_fragments
            registry = PackRegistry.discover()
            guidance_fragments = load_guidance_fragments(registry, selected_guidance=guidance_packs)
            system_prompt = assemble_system_prompt(base_prompt, guidance_fragments)
        except Exception as e:
            console.print(f"[red]ERROR: Failed to load guidance packs: {e}[/red]")
            return
    else:
        system_prompt = base_prompt

    if guidance_packs:
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        guidance_meta = {
            "base_system_prompt": base_prompt,
            "guidance_packs": guidance_packs,
            "guidance_fragments": guidance_fragments,
            "guidance_prompt_hash": prompt_hash,
        }

    selected_conditions = list(CONDITIONS)
    if getattr(args, "app", "all") != "all":
        selected_conditions = [c for c in selected_conditions if c.get("app") == args.app]

    if getattr(args, "start", "all") != "all":
        want_agent = (args.start == "agent")
        selected_conditions = [c for c in selected_conditions if ("/agent" in str(c.get("target_url", ""))) == want_agent]

    if getattr(args, "discoverability", "all") != "all":
        selected_conditions = [c for c in selected_conditions if c.get("discoverability") == args.discoverability]

    if getattr(args, "capability", "all") != "all":
        selected_conditions = [c for c in selected_conditions if c.get("capability") == args.capability]

    if not selected_conditions:
        console.print("[red]ERROR: No conditions matched the provided filters.[/red]")
        return

    # Handle custom instruction (ad-hoc task) vs predefined tasks
    custom_instruction = getattr(args, "instruction", None)
    if custom_instruction:
        # Create an ad-hoc task with no verifier (runs until agent says done or max iterations)
        from datetime import datetime
        adhoc_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tasks_to_run = [{
            "id": adhoc_id,
            "instruction": custom_instruction,
            "verifier": lambda s: False  # Never auto-pass; rely on agent's TASK_COMPLETE
        }]
        console.print(f"[cyan]Running ad-hoc task: {custom_instruction[:60]}{'...' if len(custom_instruction) > 60 else ''}[/cyan]")
    else:
        tasks_to_run = list(TASKS)
        if getattr(args, "tasks", "all") != "all":
            requested = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
            requested_set = set(requested)
            tasks_to_run = [t for t in TASKS if t.get("id") in requested_set]
            missing = requested_set - {t.get("id") for t in tasks_to_run}
            if missing:
                console.print(f"[yellow]Warning: unknown task ids ignored: {', '.join(sorted(missing))}[/yellow]")
            if not tasks_to_run:
                console.print("[red]ERROR: No tasks matched --tasks filter.[/red]")
                return

    gate_policy = _load_gate_policy(getattr(args, "gate_policy", None))
    advisor_pack = str(getattr(args, "advisor_pack", "") or "")
    advisor_id = str(getattr(args, "advisor_id", "") or "")
    if advisor_pack and not advisor_id:
        console.print("[red]ERROR: --advisor-id is required when using --advisor-pack[/red]")
        return
    if advisor_id and not advisor_pack:
        console.print("[red]ERROR: --advisor-pack is required when using --advisor-id[/red]")
        return

    runs_per_task = int(getattr(args, "runs_per_task", RUNS_PER_TASK))
    if runs_per_task < 1:
        console.print("[red]ERROR: --runs-per-task must be >= 1[/red]")
        return

    # Get model from args
    model = getattr(args, "model", "sonnet")
    model_full_name = ComputerUseAgent.SUPPORTED_MODELS.get(model, model)

    CONDITIONS = selected_conditions

    console.rule("[bold]Commerce ACI Benchmark - Claude Computer Use[/bold]")
    console.print(f"Model: {model_full_name} ({model})")
    console.print(f"Beta: computer-use-2025-01-24,prompt-caching-2024-07-31")
    console.print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    console.print(f"Max iterations: {MAX_ITERATIONS}")
    console.print(f"Runs per task: {runs_per_task}")
    console.print(f"Conditions: {len(CONDITIONS)}")
    console.print(f"Debug screenshots: {DEBUG_DIR}")
    console.print()

    for cond in CONDITIONS:
        console.rule(f"[cyan]{cond['name']}[/cyan]")

        for task in tasks_to_run:
            for run_num in range(runs_per_task):
                try:
                    res = await run_trial(
                        task,
                        cond["target_url"],
                        cond["api_url"],
                        cond["name"],
                        run_num + 1,
                        discoverability=cond["discoverability"],
                        capability=cond["capability"],
                        system_prompt=system_prompt,
                        model=model,
                        guidance_meta=guidance_meta,
                    )
                    results.append({
                        **res,
                        "condition": cond["name"],
                        "app": cond["app"],
                        "discoverability": cond["discoverability"],
                        "capability": cond["capability"],
                        "task": task["id"],
                        "run": run_num + 1
                    })

                    status = "[green]PASS[/green]" if res['success'] else "[red]FAIL[/red]"
                    console.print(f"  {task['id']} | Run {run_num + 1} | {status} | Steps: {res['steps']}")

                    eval_packs_raw = getattr(args, "eval_packs", "")
                    eval_selectors = [p.strip() for p in str(eval_packs_raw).split(",") if p.strip()]
                    eval_output = getattr(args, "eval_output", "eval_results.json")
                    trace_path_str = res.get("trace_path")
                    if eval_selectors and trace_path_str:
                        try:
                            from pack_api.loader import PackRegistry
                            from pack_api.runtime import (
                                gate_should_fail,
                                run_evaluators_on_trace,
                                suggest_guidance_patch,
                                write_eval_results,
                            )

                            trace_path = Path(trace_path_str)
                            trace = json.loads(trace_path.read_text(encoding="utf-8"))
                            registry = PackRegistry.discover()
                            eval_results = run_evaluators_on_trace(
                                trace,
                                selected_evaluators=eval_selectors,
                                registry=registry,
                            )
                            out_path = write_eval_results(
                                trace_path,
                                eval_results,
                                out_path=trace_path.parent / eval_output,
                                policy=gate_policy,
                            )
                            gate_failed = gate_should_fail(eval_results, gate_policy)
                            gate_status = "FAIL" if gate_failed else "PASS"
                            console.print(f"    Eval selectors: {', '.join(eval_selectors)} -> {out_path} ({gate_status})")

                            if gate_failed and advisor_pack and advisor_id:
                                try:
                                    advisor_spec = registry.get_advisor_spec(advisor_pack, advisor_id)
                                    advisor_fn = registry.load_advisor(advisor_pack, advisor_id)
                                    patch = suggest_guidance_patch(
                                        advisor_fn,
                                        trace,
                                        eval_results,
                                        config=advisor_spec.default_config,
                                    )
                                    if patch:
                                        patch_path = trace_path.parent / "guidance_patch.json"
                                        patch_path.write_text(json.dumps(patch.__dict__, indent=2), encoding="utf-8")
                                        console.print(f"    Guidance patch: {patch_path}")
                                    else:
                                        console.print("    Guidance patch: skipped")
                                except Exception as e:
                                    console.print(f"[yellow]Warning: advisor failed: {e}[/yellow]")
                        except Exception as e:
                            console.print(f"[yellow]Warning: evaluator run failed: {e}[/yellow]")

                except Exception as e:
                    import traceback
                    console.print(f"  {task['id']} | Run {run_num + 1} | [red]ERROR: {e}[/red]")
                    traceback.print_exc()
                    results.append({
                        "success": False,
                        "steps": 0,
                        "entered_agent_view": False,
                        "first_agent_view_step": None,
                        "agent_actions": 0,
                        "model_calls": 0,
                        "ui_actions": 0,
                        "wall_time_seconds": 0,
                        "forced_agent_start": "/agent" in cond["target_url"],
                        "condition": cond["name"],
                        "app": cond["app"],
                        "discoverability": cond["discoverability"],
                        "capability": cond["capability"],
                        "task": task["id"],
                        "run": run_num + 1,
                        "error": str(e)
                    })

    # --- METRICS REPORTING ---
    console.print()

    # Helper function to compute metrics for a subset
    def compute_metrics(subset: list) -> dict:
        if not subset:
            return None
        wins = [r for r in subset if r["success"]]
        acc = len(wins) / len(subset) * 100
        avg_steps = statistics.mean([r["steps"] for r in wins]) if wins else 0.0

        # Calculate adoption - show "N/A" for forced /agent starts
        forced_starts = [r for r in subset if r.get("forced_agent_start", False)]
        if len(forced_starts) == len(subset):
            adoption_str = "N/A"
            avg_adoption_step = None
        else:
            non_forced = [r for r in subset if not r.get("forced_agent_start", False)]
            if non_forced:
                adopters = [r for r in non_forced if r["entered_agent_view"] or r["agent_actions"] > 0]
                adoption = (len(adopters) / len(non_forced)) * 100
                adoption_str = f"{adoption:.0f}%"
                # Calculate average step when adoption occurred
                adoption_steps = [r.get("first_agent_view_step") for r in adopters
                                  if r.get("first_agent_view_step") is not None]
                avg_adoption_step = statistics.mean(adoption_steps) if adoption_steps else None
            else:
                adoption_str = "N/A"
                avg_adoption_step = None

        # Efficiency metrics are wins-only (comparing efficiency on successful runs)
        avg_model_calls = statistics.mean([r.get("model_calls", 0) for r in wins]) if wins else 0.0
        avg_ui_actions = statistics.mean([r.get("ui_actions", 0) for r in wins]) if wins else 0.0
        avg_wall_time = statistics.mean([r.get("wall_time_seconds", 0) for r in wins]) if wins else 0.0

        return {
            "n": len(subset),
            "n_wins": len(wins),
            "accuracy": acc,
            "avg_steps": avg_steps,
            "avg_model_calls": avg_model_calls,
            "avg_ui_actions": avg_ui_actions,
            "avg_wall_time": avg_wall_time,
            "adoption": adoption_str,
            "avg_adoption_step": avg_adoption_step,
            "avg_agent_actions": statistics.mean([r["agent_actions"] for r in wins]) if wins else 0.0
        }

    # === TABLE 1: Full condition breakdown ===
    table = Table(title="Results by Condition")
    table.add_column("Condition")
    table.add_column("Accuracy")
    table.add_column("Steps")
    table.add_column("Model")
    table.add_column("UI")
    table.add_column("Time")
    table.add_column("Adopt")
    table.add_column("AgentAPI")

    for cond in CONDITIONS:
        name = cond["name"]
        subset = [r for r in results if r["condition"] == name]
        m = compute_metrics(subset)
        if m:
            table.add_row(
                name,
                f"{m['accuracy']:.0f}%",
                f"{m['avg_steps']:.1f}",
                f"{m['avg_model_calls']:.1f}",
                f"{m['avg_ui_actions']:.1f}",
                f"{m['avg_wall_time']:.1f}s",
                m['adoption'],
                f"{m['avg_agent_actions']:.1f}"
            )
    console.print(table)

    # === TABLE 2: Factor split by App ===
    console.print()
    app_table = Table(title="Results by App")
    app_table.add_column("App")
    app_table.add_column("N")
    app_table.add_column("Accuracy")
    app_table.add_column("Steps")
    app_table.add_column("Time")

    for app in ["baseline", "treatment", "treatment-docs"]:
        subset = [r for r in results if r.get("app") == app]
        m = compute_metrics(subset)
        if m:
            app_table.add_row(
                app,
                str(m['n']),
                f"{m['accuracy']:.0f}%",
                f"{m['avg_steps']:.1f}",
                f"{m['avg_wall_time']:.1f}s"
            )
    console.print(app_table)

    # === TABLE 3: Factor split by Capability ===
    console.print()
    cap_table = Table(title="Results by Capability (treatment apps only)")
    cap_table.add_column("Capability")
    cap_table.add_column("N")
    cap_table.add_column("Accuracy")
    cap_table.add_column("Steps")
    cap_table.add_column("AgentAPI")

    for cap in ["advantage", "parity"]:
        subset = [r for r in results if r.get("capability") == cap and r.get("app") != "baseline"]
        m = compute_metrics(subset)
        if m:
            cap_table.add_row(
                cap,
                str(m['n']),
                f"{m['accuracy']:.0f}%",
                f"{m['avg_steps']:.1f}",
                f"{m['avg_agent_actions']:.1f}"
            )
    console.print(cap_table)

    # === TABLE 4: Factor split by Discoverability ===
    console.print()
    disc_table = Table(title="Results by Discoverability (root starts only)")
    disc_table.add_column("Discoverability")
    disc_table.add_column("N")
    disc_table.add_column("Accuracy")
    disc_table.add_column("Adoption")
    disc_table.add_column("Adopt@Step")

    for disc in ["navbar", "hidden"]:
        # Only look at root starts (non-forced) for discoverability analysis
        subset = [r for r in results
                  if r.get("discoverability") == disc
                  and not r.get("forced_agent_start", False)
                  and r.get("app") != "baseline"]
        m = compute_metrics(subset)
        if m:
            adopt_step_str = f"{m['avg_adoption_step']:.1f}" if m['avg_adoption_step'] else "N/A"
            disc_table.add_row(
                disc,
                str(m['n']),
                f"{m['accuracy']:.0f}%",
                m['adoption'],
                adopt_step_str
            )
    console.print(disc_table)

    # === TABLE 5: Conditional success (discovery runs) ===
    # For runs starting at root: compare success rate when agent adopted vs didn't adopt
    console.print()
    cond_table = Table(title="Conditional Success (discovery runs, treatment apps)")
    cond_table.add_column("Adopted Agent UI?")
    cond_table.add_column("N")
    cond_table.add_column("Success Rate")
    cond_table.add_column("Avg Steps (wins)")

    # Only look at discovery runs (root starts, not baseline)
    discovery_runs = [r for r in results
                      if not r.get("forced_agent_start", False)
                      and r.get("app") != "baseline"]

    if discovery_runs:
        # Split by adoption
        adopted = [r for r in discovery_runs if r.get("entered_agent_view") or r.get("agent_actions", 0) > 0]
        not_adopted = [r for r in discovery_runs if not (r.get("entered_agent_view") or r.get("agent_actions", 0) > 0)]

        for label, subset in [("Yes", adopted), ("No", not_adopted)]:
            if subset:
                wins = [r for r in subset if r["success"]]
                success_rate = len(wins) / len(subset) * 100
                avg_steps = statistics.mean([r["steps"] for r in wins]) if wins else 0.0
                cond_table.add_row(
                    label,
                    str(len(subset)),
                    f"{success_rate:.0f}%",
                    f"{avg_steps:.1f}"
                )
    console.print(cond_table)

    # Save results
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"computeruse_{run_id}.json"

    # Compute factor-level aggregates for JSON output
    factor_aggregates = {
        "by_app": {},
        "by_capability": {},
        "by_discoverability": {}
    }

    for app in ["baseline", "treatment", "treatment-docs"]:
        subset = [r for r in results if r.get("app") == app]
        m = compute_metrics(subset)
        if m:
            factor_aggregates["by_app"][app] = m

    for cap in ["advantage", "parity"]:
        subset = [r for r in results if r.get("capability") == cap and r.get("app") != "baseline"]
        m = compute_metrics(subset)
        if m:
            factor_aggregates["by_capability"][cap] = m

    for disc in ["navbar", "hidden"]:
        subset = [r for r in results
                  if r.get("discoverability") == disc
                  and not r.get("forced_agent_start", False)
                  and r.get("app") != "baseline"]
        m = compute_metrics(subset)
        if m:
            factor_aggregates["by_discoverability"][disc] = m

    output_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_full_name,
        "beta": "computer-use-2025-01-24,prompt-caching-2024-07-31",
        "config": {
            "display_width": DISPLAY_WIDTH,
            "display_height": DISPLAY_HEIGHT,
            "max_iterations": MAX_ITERATIONS,
            "runs_per_task": runs_per_task,
            "treatment_url": URL_TREATMENT,
            "treatment_docs_url": URL_TREATMENT_DOCS,
            "baseline_url": URL_BASELINE,
            "conditions": [c["name"] for c in CONDITIONS]
        },
        "factors": {
            "apps": ["baseline", "treatment", "treatment-docs"],
            "discoverability": ["navbar", "hidden"],
            "capability": ["advantage", "parity"]
        },
        "factor_aggregates": factor_aggregates,
        "results": results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")
    console.print(f"[green]Debug screenshots saved to: {DEBUG_DIR}[/green]")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commerce ACI Benchmark - Claude Computer Use")
    parser.add_argument("--app", choices=["all", "baseline", "treatment", "treatment-docs"], default="all",
                        help="Which app/UI to run (default: all)")
    parser.add_argument("--start", choices=["all", "root", "agent"], default="all",
                        help="Entry point: root or /agent (default: all)")
    parser.add_argument("--discoverability", choices=["all", "navbar", "hidden"], default="all",
                        help="Agent UI link discoverability (treatment apps only)")
    parser.add_argument("--capability", choices=["all", "advantage", "parity"], default="all",
                        help="Agent capability mode (treatment apps only)")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task ids to run (default: all)")
    parser.add_argument("--runs-per-task", type=int, default=RUNS_PER_TASK,
                        help="Number of runs per task (default: RUNS_PER_TASK constant)")
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Optional path to a system prompt override (applies to the agent policy)")
    parser.add_argument(
        "--guidance-packs",
        type=str,
        default="",
        help="Comma-separated selectors: pack_id or pack_id:guidance_id (use fully-qualified IDs when in doubt)",
    )
    parser.add_argument("--instruction", type=str, default=None,
                        help="Custom instruction/goal for ad-hoc runs (overrides --tasks)")
    parser.add_argument("--model", type=str, choices=["sonnet", "haiku"], default="sonnet",
                        help="Model to use: sonnet (claude-sonnet-4-5) or haiku (claude-haiku-3-5). Default: sonnet")
    parser.add_argument(
        "--eval-packs",
        type=str,
        default="",
        help="Comma-separated selectors: pack_id or pack_id:evaluator_id (use fully-qualified IDs when in doubt)",
    )
    parser.add_argument("--eval-output", type=str, default="eval_results.json",
                        help="Eval results filename to write next to trace.json")
    parser.add_argument("--gate-policy", type=str, default="",
                        help="Optional TOML policy file for evaluator gating")
    parser.add_argument("--advisor-pack", type=str, default="",
                        help="Optional advisor pack ID to suggest guidance patches on gate failure")
    parser.add_argument("--advisor-id", type=str, default="",
                        help="Advisor ID within the advisor pack")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
