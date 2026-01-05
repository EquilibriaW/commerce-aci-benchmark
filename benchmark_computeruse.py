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
URL_BASELINE = os.getenv("URL_BASELINE", "http://localhost:3001")

BENCHMARK_SECRET = os.getenv("BENCHMARK_SECRET", "sk-bench-123")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Computer use display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800

MAX_ITERATIONS = 15
RUNS_PER_TASK = 1  # Reduced for testing / rate limits

console = Console()

# --- VERIFIABLE TASK SUITE ---
# Benchmark products:
# - black-t-shirt: $20.00, sizes S/M/L
# - acme-cup: $15.00, no variants
# - hoodie: $50.00, no variants

TASKS = [
    {
        "id": "t01_find_add_simple",
        "instruction": "Find the Black T-Shirt product and add 1 unit to the cart.",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and i['quantity'] >= 1
            for i in s['cart']['items']
        )
    },
    {
        "id": "t02_variant_size_l",
        "instruction": "Find the Black T-Shirt, select Size L, and add it to the cart.",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and i.get('variant') == 'L'
            for i in s['cart']['items']
        )
    },
    {
        "id": "t03_cart_total_check",
        "instruction": "Add 2 Acme Cup ($15 each) and 1 Hoodie ($50) to the cart. The total should be $80.",
        "verifier": lambda s: s['cart']['total_price_cents'] == 8000
    }
]


class ComputerUseAgent:
    """Agent that uses Claude's computer use capability with screenshots."""

    def __init__(self, page: Page, api_key: str):
        self.page = page
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        self.entered_agent_view = False
        self.agent_action_requests = 0

        # Track network requests to /agent/actions
        self.page.on("request", self._handle_request)

    def _handle_request(self, request):
        if request.method == "POST" and "/agent/actions/" in request.url:
            self.agent_action_requests += 1

    async def take_screenshot(self) -> str:
        """Take a screenshot and return as base64."""
        screenshot_bytes = await self.page.screenshot(type="png")
        return base64.standard_b64encode(screenshot_bytes).decode("utf-8")

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

    async def run_step(self, instruction: str) -> tuple[str, bool]:
        """
        Run one step of the agent loop.
        Returns (action_taken, is_done).
        """
        # Check if we're in agent view
        if "/agent" in self.page.url:
            self.entered_agent_view = True

        # Take screenshot
        screenshot_b64 = await self.take_screenshot()

        # Build messages
        if not self.conversation_history:
            # First message - include the task instruction
            self.conversation_history = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an AI agent tasked with completing an e-commerce task.

TASK: {instruction}

Use the computer tool to interact with the webpage. You can:
- Click on elements using coordinates
- Type text
- Scroll the page
- Take screenshots

Look at the current screenshot and decide what action to take next.
When you have completed the task, respond with "TASK_COMPLETE" in your message."""
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
            ]
        else:
            # Follow-up message with new screenshot
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

        # Call Claude with computer use (beta API) - with retry for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=4096,
                    extra_headers={"anthropic-beta": "computer-use-2025-11-24"},
                    tools=[
                        {
                            "type": "computer_20251124",
                            "name": "computer",
                            "display_width_px": DISPLAY_WIDTH,
                            "display_height_px": DISPLAY_HEIGHT,
                            "display_number": 1
                        }
                    ],
                    messages=self.conversation_history
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f"    Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        # Process response
        assistant_content = []
        action_taken = "none"
        is_done = False

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                if "TASK_COMPLETE" in block.text.upper():
                    is_done = True
                    action_taken = "TASK_COMPLETE"

            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

                # Execute the computer action
                tool_result = await self._execute_computer_action(block.input)
                action_taken = f"{block.input.get('action', 'unknown')}"

                # Add assistant message and tool result
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result
                        }
                    ]
                })
                return action_taken, is_done

        # No tool use - just text response
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_content
        })

        return action_taken, is_done

    async def _execute_computer_action(self, action_input: dict) -> str:
        """Execute a computer use action and return result."""
        action = action_input.get("action")

        try:
            if action == "screenshot":
                return "Screenshot taken successfully."

            elif action == "mouse_move":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.move(x, y)
                return f"Moved mouse to ({x}, {y})"

            elif action == "left_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.click(x, y)
                await asyncio.sleep(0.5)  # Wait for any navigation/updates
                return f"Left clicked at ({x}, {y})"

            elif action == "left_click_drag":
                start = action_input.get("start_coordinate", [0, 0])
                end = action_input.get("coordinate", [0, 0])
                await self.page.mouse.move(start[0], start[1])
                await self.page.mouse.down()
                await self.page.mouse.move(end[0], end[1])
                await self.page.mouse.up()
                return f"Dragged from {start} to {end}"

            elif action == "right_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.click(x, y, button="right")
                return f"Right clicked at ({x}, {y})"

            elif action == "double_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.dblclick(x, y)
                return f"Double clicked at ({x}, {y})"

            elif action == "triple_click":
                x = action_input.get("coordinate", [0, 0])[0]
                y = action_input.get("coordinate", [0, 0])[1]
                await self.page.mouse.click(x, y, click_count=3)
                return f"Triple clicked at ({x}, {y})"

            elif action == "scroll":
                x = action_input.get("coordinate", [DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2])[0]
                y = action_input.get("coordinate", [DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2])[1]
                delta_x = action_input.get("delta_x", 0)
                delta_y = action_input.get("delta_y", 0)
                await self.page.mouse.move(x, y)
                await self.page.mouse.wheel(delta_x, delta_y)
                return f"Scrolled at ({x}, {y}) by delta ({delta_x}, {delta_y})"

            elif action == "type":
                text = action_input.get("text", "")
                await self.page.keyboard.type(text)
                return f"Typed: {text[:50]}..."

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
                return f"Pressed key: {key}"

            elif action == "hold_key":
                key = action_input.get("key", "")
                await self.page.keyboard.down(key)
                return f"Holding key: {key}"

            elif action == "release_key":
                key = action_input.get("key", "")
                await self.page.keyboard.up(key)
                return f"Released key: {key}"

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Action failed: {str(e)}"


async def run_trial(task: dict, target_url: str, api_url: str) -> dict:
    """Run a single trial of a task."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

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
        agent = ComputerUseAgent(page, ANTHROPIC_API_KEY)

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

        await browser.close()

        return {
            "success": success,
            "steps": steps,
            "entered_agent_view": agent.entered_agent_view,
            "agent_actions": agent.agent_action_requests
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
            "name": "Treatment (Agent UI)",
            "target_url": f"{URL_TREATMENT}/agent",
            "api_url": URL_TREATMENT
        },
        {
            "name": "Discovery (Root)",
            "target_url": URL_TREATMENT,
            "api_url": URL_TREATMENT
        }
    ]

    console.rule("[bold]Commerce ACI Benchmark - Claude Computer Use[/bold]")
    console.print(f"Model: claude-opus-4-5-20251101")
    console.print(f"Beta: computer-use-2025-11-24")
    console.print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    console.print(f"Max iterations: {MAX_ITERATIONS}")
    console.print(f"Runs per task: {RUNS_PER_TASK}")
    console.print()

    for cond in CONDITIONS:
        console.rule(f"[cyan]{cond['name']}[/cyan]")

        for task in TASKS:
            for run_num in range(RUNS_PER_TASK):
                try:
                    res = await run_trial(task, cond["target_url"], cond["api_url"])
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

        adopters = [r for r in subset if r["entered_agent_view"] or r["agent_actions"] > 0]
        adoption = (len(adopters) / len(subset)) * 100

        avg_actions = statistics.mean([r["agent_actions"] for r in subset])

        table.add_row(
            name,
            f"{acc:.0f}%",
            f"{avg_steps:.1f}",
            f"{adoption:.0f}%",
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
        "model": "claude-opus-4-5-20251101",
        "beta": "computer-use-2025-11-24",
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


if __name__ == "__main__":
    asyncio.run(main())
