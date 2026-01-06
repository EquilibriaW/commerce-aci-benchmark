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
import json
import statistics
import time
from datetime import datetime

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
from pathlib import Path
from typing import Optional

import httpx
from anthropic import Anthropic
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
                  'left_click_drag', 'type', 'key'}
    # Max conversation cycles to keep (preserve initial prompt, truncate older cycles)
    MAX_HISTORY_CYCLES = 5

    def __init__(self, page: Page, api_key: str, debug_dir: Optional[Path] = None):
        self.page = page
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        self.initial_prompt = None  # Store the initial user prompt (preserved across truncation)
        self.entered_agent_view = False
        self.agent_action_requests = 0
        self.step_count = 0
        self.model_calls = 0  # Number of LLM API calls
        self.ui_actions = 0   # Number of real UI actions (clicks/drag/type/key)
        self.debug_dir = debug_dir
        self.action_log = []

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

    def _save_debug_screenshot(self, screenshot_bytes: bytes, step: int):
        """Save screenshot to debug directory."""
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = self.debug_dir / f"step_{step:02d}.png"
            screenshot_path.write_bytes(screenshot_bytes)

    def _log_action(self, step: int, action: str, details: str):
        """Log an action for debugging."""
        log_entry = f"Step {step:02d}: {action} - {details}"
        self.action_log.append(log_entry)
        console.print(f"    [dim]{log_entry}[/dim]")

    async def reset_session(self, base_url: str):
        """Reset the session via API."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/agent/reset",
                headers={"X-Benchmark-Secret": BENCHMARK_SECRET}
            )
            if resp.status_code != 200:
                raise Exception(f"Failed to reset session: {resp.status_code} {resp.text}")

            # Extract and set the cookie
            if "set-cookie" in resp.headers:
                cookie_header = resp.headers["set-cookie"]
                # Parse cartId from cookie
                for part in cookie_header.split(";"):
                    if "cartId=" in part:
                        cart_id = part.split("cartId=")[1].strip()
                        await self.page.context.add_cookies([{
                            "name": "cartId",
                            "value": cart_id,
                            "domain": "localhost",
                            "path": "/"
                        }])
                        break

    async def get_ground_truth(self, base_url: str) -> dict:
        """Get the current cart state from the API."""
        # Get cookies from browser context
        cookies = await self.page.context.cookies()
        cookie_header = "; ".join([f"{c['name']}={c['value']}" for c in cookies])

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{base_url}/agent/state",
                headers={
                    "X-Benchmark-Secret": BENCHMARK_SECRET,
                    "Cookie": cookie_header
                }
            )
            if resp.status_code == 200:
                return resp.json()
            return {}

    def _truncate_history(self):
        """Truncate conversation history while preserving initial prompt.

        Keeps the initial user message (with task instruction) and the last
        MAX_HISTORY_CYCLES complete interaction cycles. Only truncates whole
        assistant+user pairs to avoid breaking tool_use/tool_result pairings.
        """
        if len(self.conversation_history) <= 1:
            return

        messages_after_initial = self.conversation_history[1:]

        # We want to keep at most MAX_HISTORY_CYCLES * 2 messages after initial
        max_messages = self.MAX_HISTORY_CYCLES * 2

        if len(messages_after_initial) <= max_messages:
            return

        # Find how many to remove (must be even to keep assistant+user pairs intact)
        to_remove = len(messages_after_initial) - max_messages
        to_remove = (to_remove // 2) * 2  # Round down to even

        if to_remove > 0:
            remaining = messages_after_initial[to_remove:]

            # Safety check: ensure we don't start with a user message containing tool_result
            # (which would be orphaned without the preceding assistant tool_use)
            while remaining and remaining[0].get("role") == "user":
                content = remaining[0].get("content", [])
                has_tool_result = any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in (content if isinstance(content, list) else [])
                )
                if has_tool_result:
                    # Remove this orphaned tool_result message
                    remaining = remaining[1:]
                else:
                    break

            self.conversation_history = [self.conversation_history[0]] + remaining

    async def run_step(self, instruction: str) -> tuple[str, bool]:
        """
        Run one step of the agent loop.
        Returns (action_taken, is_done).
        """
        self.step_count += 1

        # Check if we're in agent view
        if "/agent" in self.page.url:
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

        if not last_was_tool_result:
            # Take screenshot
            screenshot_b64, screenshot_bytes = await self.take_screenshot()

            # Save debug screenshot
            self._save_debug_screenshot(screenshot_bytes, self.step_count)

            # Build messages - use system parameter for system prompt
            if not self.conversation_history:
                # First message - task instruction + screenshot
                self.initial_prompt = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"TASK: {instruction}\n\nHere is the current state of the webpage:"
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
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the current state of the page after your last action. Continue with the task or say TASK_COMPLETE if done."
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
        response = None
        while response is None:
            try:
                self.model_calls += 1
                response = self.client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    extra_headers={"anthropic-beta": "computer-use-2025-01-24"},
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
            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit" in error_str:
                    console.print("    [yellow]Rate limited, waiting 60s...[/yellow]")
                    await asyncio.sleep(60)
                elif "500" in str(e) or "internal" in error_str:
                    console.print("    [yellow]API 500 error, retrying in 10s...[/yellow]")
                    await asyncio.sleep(10)
                else:
                    raise

        # Process response - handle multiple tool_use blocks
        assistant_content = []
        tool_uses = []
        action_taken = "none"
        is_done = False

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
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

        # Add assistant message first (contains all tool_use blocks)
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })

        # Execute all tool_uses sequentially and collect results
        if tool_uses:
            tool_results = []
            for block in tool_uses:
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

                # Execute the computer action
                tool_result = await self._execute_computer_action(block.input)
                action_taken = action_type

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

        return action_taken, is_done

    async def _execute_computer_action(self, action_input: dict) -> list:
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
                await self.page.keyboard.press(key)
                result_text = f"Pressed key: {key}"

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
        screenshot_b64, _ = await self.take_screenshot()

        return [
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


async def run_trial(task: dict, target_url: str, api_url: str, condition_name: str, run_num: int) -> dict:
    """Run a single trial of a task."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Track if we're starting on /agent (forced adoption - show N/A for adoption metric)
    forced_agent_start = "/agent" in target_url

    # Set up debug directory
    debug_dir = None
    if DEBUG_SCREENSHOTS:
        # Sanitize condition name for directory
        safe_condition = condition_name.replace(" ", "_").replace("(", "").replace(")", "")
        debug_dir = DEBUG_DIR / safe_condition / task["id"] / f"run_{run_num:02d}"

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
        agent = ComputerUseAgent(page, ANTHROPIC_API_KEY, debug_dir=debug_dir)

        # Reset session
        await agent.reset_session(api_url)

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

        # End wall time tracking
        wall_time_seconds = time.perf_counter() - start_time

        # Save action log
        if debug_dir:
            log_path = debug_dir / "actions.log"
            log_path.write_text("\n".join(agent.action_log))

        await browser.close()

        return {
            "success": success,
            "steps": steps,
            "entered_agent_view": agent.entered_agent_view,
            "agent_actions": agent.agent_action_requests,
            "model_calls": agent.model_calls,
            "ui_actions": agent.ui_actions,
            "wall_time_seconds": round(wall_time_seconds, 2),
            "forced_agent_start": forced_agent_start,
            "action_log": agent.action_log
        }


async def main():
    """Run the full benchmark."""
    if not ANTHROPIC_API_KEY:
        console.print("[red]ERROR: ANTHROPIC_API_KEY environment variable is required[/red]")
        return

    results = []

    # Test conditions
    CONDITIONS = [
        {
            "name": "Control (Human UI)",
            "target_url": URL_BASELINE,
            "api_url": URL_BASELINE
        },
        {
            "name": "Treatment 1 (Terminal UI)",
            "target_url": f"{URL_TREATMENT}/agent",
            "api_url": URL_TREATMENT
        },
        {
            "name": "Discovery 1 (Terminal Root)",
            "target_url": URL_TREATMENT,
            "api_url": URL_TREATMENT
        },
        {
            "name": "Treatment 2 (Doc UI)",
            "target_url": f"{URL_TREATMENT_DOCS}/agent",
            "api_url": URL_TREATMENT_DOCS
        },
        {
            "name": "Discovery 2 (Doc Root)",
            "target_url": URL_TREATMENT_DOCS,
            "api_url": URL_TREATMENT_DOCS
        }
    ]

    console.rule("[bold]Commerce ACI Benchmark - Claude Computer Use[/bold]")
    console.print(f"Model: claude-sonnet-4-5-20250929")
    console.print(f"Beta: computer-use-2025-01-24")
    console.print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    console.print(f"Max iterations: {MAX_ITERATIONS}")
    console.print(f"Runs per task: {RUNS_PER_TASK}")
    console.print(f"Debug screenshots: {DEBUG_DIR}")
    console.print()

    for cond in CONDITIONS:
        console.rule(f"[cyan]{cond['name']}[/cyan]")

        for task in TASKS:
            for run_num in range(RUNS_PER_TASK):
                try:
                    res = await run_trial(
                        task,
                        cond["target_url"],
                        cond["api_url"],
                        cond["name"],
                        run_num + 1
                    )
                    results.append({
                        **res,
                        "condition": cond["name"],
                        "task": task["id"],
                        "run": run_num + 1
                    })

                    status = "[green]PASS[/green]" if res['success'] else "[red]FAIL[/red]"
                    console.print(f"  {task['id']} | Run {run_num + 1} | {status} | Steps: {res['steps']}")

                except Exception as e:
                    import traceback
                    console.print(f"  {task['id']} | Run {run_num + 1} | [red]ERROR: {e}[/red]")
                    traceback.print_exc()
                    results.append({
                        "success": False,
                        "steps": 0,
                        "entered_agent_view": False,
                        "agent_actions": 0,
                        "model_calls": 0,
                        "ui_actions": 0,
                        "wall_time_seconds": 0,
                        "forced_agent_start": "/agent" in cond["target_url"],
                        "condition": cond["name"],
                        "task": task["id"],
                        "run": run_num + 1,
                        "error": str(e)
                    })

    # --- METRICS REPORTING ---
    console.print()
    table = Table(title="Benchmark Results")
    table.add_column("Condition")
    table.add_column("Accuracy")
    table.add_column("Avg Steps")
    table.add_column("Model Calls")
    table.add_column("UI Actions")
    table.add_column("Wall Time")
    table.add_column("Adoption %")
    table.add_column("Agent Actions")

    for cond in CONDITIONS:
        name = cond["name"]
        subset = [r for r in results if r["condition"] == name]
        if not subset:
            continue

        wins = [r for r in subset if r["success"]]
        acc = len(wins) / len(subset) * 100
        avg_steps = statistics.mean([r["steps"] for r in wins]) if wins else 0.0

        # Calculate adoption - show "N/A" for forced /agent starts
        # (adoption is only meaningful when starting from root)
        forced_starts = [r for r in subset if r.get("forced_agent_start", False)]
        if len(forced_starts) == len(subset):
            adoption_str = "N/A"
        else:
            non_forced = [r for r in subset if not r.get("forced_agent_start", False)]
            if non_forced:
                adopters = [r for r in non_forced if r["entered_agent_view"] or r["agent_actions"] > 0]
                adoption = (len(adopters) / len(non_forced)) * 100
                adoption_str = f"{adoption:.0f}%"
            else:
                adoption_str = "N/A"

        avg_actions = statistics.mean([r["agent_actions"] for r in subset])
        avg_model_calls = statistics.mean([r.get("model_calls", 0) for r in subset])
        avg_ui_actions = statistics.mean([r.get("ui_actions", 0) for r in subset])
        avg_wall_time = statistics.mean([r.get("wall_time_seconds", 0) for r in subset])

        table.add_row(
            name,
            f"{acc:.0f}%",
            f"{avg_steps:.1f}",
            f"{avg_model_calls:.1f}",
            f"{avg_ui_actions:.1f}",
            f"{avg_wall_time:.1f}s",
            adoption_str,
            f"{avg_actions:.1f}"
        )

    console.print(table)

    # Save results
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"computeruse_{run_id}.json"

    output_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": "claude-sonnet-4-5-20250929",
        "beta": "computer-use-2025-01-24",
        "config": {
            "display_width": DISPLAY_WIDTH,
            "display_height": DISPLAY_HEIGHT,
            "max_iterations": MAX_ITERATIONS,
            "runs_per_task": RUNS_PER_TASK,
            "treatment_url": URL_TREATMENT,
            "baseline_url": URL_BASELINE
        },
        "results": results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")
    console.print(f"[green]Debug screenshots saved to: {DEBUG_DIR}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
