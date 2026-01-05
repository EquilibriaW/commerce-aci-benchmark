import asyncio
import os
import json
import statistics
from typing import Literal, Optional, Set
from pydantic import BaseModel, Field, ConfigDict
from playwright.async_api import async_playwright, Page
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

# --- CONFIGURATION ---
# Update these to match your running ports if different
URL_TREATMENT = "http://localhost:3000"
URL_BASELINE = "http://localhost:3001"

BENCHMARK_SECRET = os.getenv("BENCHMARK_SECRET", "sk-bench-123")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_ITERATIONS = 15
RUNS_PER_TASK = 5

# --- STRICT SCHEMA DEFINITION ---
class AgentAction(BaseModel):
    # Forbids the LLM from adding hallucinated fields
    model_config = ConfigDict(extra="forbid") 
    
    action_type: Literal["CLICK", "TYPE", "GOTO", "DONE"]
    target_type: Literal["agent_id", "css", "text"]
    target: str
    value: Optional[str] = None
    reasoning: Optional[str] = None

# --- VERIFIABLE TASK SUITE ---
# Products available:
# - classic-leather-jacket: $299, sizes S/M/L/XL
# - minimalist-watch: $199, colors Black/Silver/Gold
# - organic-cotton-tee: $45, variants S/White, M/White, L/Black
# - wool-blend-scarf: $85, colors Charcoal/Burgundy
# - canvas-sneakers: $79, sizes 9/10 in White/Black

TASKS = [
    {
        "id": "t01_find_add_simple",
        "instruction": "Find the 'Classic Leather Jacket' and add 1 unit to the cart.",
        "verifier": lambda s: any(i['slug'] == 'classic-leather-jacket' and i['quantity'] == 1 for i in s['cart']['items'])
    },
    {
        "id": "t02_variant_size_l",
        "instruction": "Find the 'Classic Leather Jacket', select Size 'L', and add to cart.",
        "verifier": lambda s: any(i['slug'] == 'classic-leather-jacket' and i.get('variant') == 'L' for i in s['cart']['items'])
    },
    {
        "id": "t03_cart_total_check",
        "instruction": "Add 2 'Minimalist Watch' ($199 ea) and 1 'Wool Blend Scarf' ($85). Verify total is $483.",
        "verifier": lambda s: abs(s['cart']['total_price'] - 483.0) < 0.01
    }
]

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
console = Console()

class ScientificAgent:
    def __init__(self, page: Page):
        self.page = page
        self.entered_agent_view = False 
        self.agent_action_requests = 0 
        self.fingerprints: Set[str] = set() 

        # Event Listener: Count actual Tool usage at network layer
        self.page.on("request", self._handle_request)

    def _handle_request(self, request):
        if request.method == "POST" and "/agent/actions/" in request.url:
            self.agent_action_requests += 1

    async def reset_session(self, base_url):
        # Use context.request to ensure cookie jar consistency
        try:
            resp = await self.page.context.request.post(
                f"{base_url}/agent/reset",
                headers={"X-Benchmark-Secret": BENCHMARK_SECRET}
            )
            if not resp.ok:
                body = await resp.text()
                print(f"RESET FAILED: {base_url} returned {resp.status}: {body[:200]}")
                raise Exception(f"Failed to reset session at {base_url}: HTTP {resp.status}")
        except Exception as e:
            print(f"CRITICAL CONNECTION ERROR at {base_url}: {e}")
            raise e

    async def get_ground_truth(self, base_url) -> dict:
        try:
            resp = await self.page.context.request.get(
                f"{base_url}/agent/state",
                headers={"X-Benchmark-Secret": BENCHMARK_SECRET}
            )
            return await resp.json() if resp.ok else {}
        except: return {}

    async def get_observation(self) -> str:
        # Behavioral Adoption Check
        if "/agent" in self.page.url:
            self.entered_agent_view = True
            return await self.page.evaluate("document.body.innerText")
        
        # Human View: Simulated Accessibility Tree
        return await self.page.evaluate("""() => {
            const clone = document.body.cloneNode(true);
            const scripts = clone.querySelectorAll('script, style, svg, noscript');
            scripts.forEach(el => el.remove());
            return clone.innerText.substring(0, 10000); 
        }""")

    async def decide(self, instruction: str, step: int) -> AgentAction:
        obs = await self.get_observation()
        try:
            completion = await client.beta.chat.completions.parse(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": f"Goal: {instruction}\n\nOBSERVATION:\n{obs}"},
                    {"role": "user", "content": f"Step {step}. Decide next move."}
                ],
                response_format=AgentAction,
                temperature=0.0
            )
            
            if completion.system_fingerprint:
                self.fingerprints.add(completion.system_fingerprint)
                
            return completion.choices[0].message.parsed
        except Exception as e:
            return AgentAction(
                action_type="DONE", 
                target_type="text", 
                target="ERROR", 
                reasoning=f"API Error: {str(e)}"
            )

    async def execute(self, cmd: AgentAction) -> str:
        try:
            # Selector Logic
            sel = f"[data-agent-id='{cmd.target}']" if cmd.target_type == "agent_id" else cmd.target
            
            if cmd.action_type == "CLICK":
                if cmd.target_type == "text":
                     await self.page.get_by_text(cmd.target, exact=False).first.click(timeout=2000)
                else:
                     await self.page.locator(sel).first.click(timeout=2000)

            elif cmd.action_type == "TYPE":
                if cmd.target_type == "agent_id" or cmd.target_type == "css":
                    await self.page.fill(sel, cmd.value or "", timeout=2000)
                else:
                    await self.page.get_by_text(cmd.target).first.click()
                    await self.page.keyboard.type(cmd.value or "")

            elif cmd.action_type == "GOTO":
                await self.page.goto(cmd.target)

            elif cmd.action_type == "DONE":
                return "DONE"

            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=3000)
            except: pass
            
            return "OK"
        except Exception as e:
            return f"FAIL: {str(e)}"

async def run_trial(task, target_url, base_api_url, block_agent_routes=False):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Block Service Workers to prevent routing leaks
        context = await browser.new_context(
            user_agent="ScienceBot/1.0",
            service_workers="block" 
        )
        
        # Explicit Route Blocking for Control Group
        if block_agent_routes:
            await context.route("**/agent", lambda route: route.abort())
            await context.route("**/agent/**", lambda route: route.abort())
            
        page = await context.new_page()
        agent = ScientificAgent(page)
        
        # Reset session using the API of the target environment
        await agent.reset_session(base_api_url)
        await page.goto(target_url)
        
        steps = 0
        success = False
        
        for _ in range(MAX_ITERATIONS):
            # Verify using the API of the target environment
            state = await agent.get_ground_truth(base_api_url)
            if state and task["verifier"](state):
                success = True
                break
            
            cmd = await agent.decide(task["instruction"], steps + 1)
            
            if cmd.action_type == "DONE":
                state = await agent.get_ground_truth(base_api_url)
                if state and task["verifier"](state):
                    success = True
                break
            
            steps += 1
            await agent.execute(cmd)
            
        await browser.close()
        
        return {
            "success": success,
            "steps": steps,
            "entered_agent_view": agent.entered_agent_view,
            "agent_actions": agent.agent_action_requests,
            "fingerprints": list(agent.fingerprints)
        }

async def main():
    results = []
    
    # Updated Conditions for Multi-Port Setup
    CONDITIONS = [
        {
            "name": "Control (Human UI)", 
            "target_url": URL_BASELINE,
            "api_url": URL_BASELINE,
            "block_agent": False 
        },
        {
            "name": "Treatment (Agent UI)", 
            "target_url": f"{URL_TREATMENT}/agent",
            "api_url": URL_TREATMENT,
            "block_agent": False 
        },
        {
            "name": "Discovery (SEO)", 
            "target_url": URL_TREATMENT, # Start at root
            "api_url": URL_TREATMENT,
            "block_agent": False 
        }
    ]
    
    console.rule("[bold]Starting Hardened Benchmark[/bold]")
    
    for cond in CONDITIONS:
        for task in TASKS:
            for i in range(RUNS_PER_TASK):
                res = await run_trial(
                    task, 
                    cond["target_url"], 
                    cond["api_url"], 
                    cond["block_agent"]
                )
                results.append({**res, "condition": cond["name"], "task": task["id"]})
                
                status = "[green]PASS[/green]" if res['success'] else "[red]FAIL[/red]"
                print(f"{cond['name']} | {task['id']} | T{i+1} | {status}")

    # --- METRICS REPORTING ---
    table = Table(title="Benchmark Metrics")
    table.add_column("Condition")
    table.add_column("Acc %")
    table.add_column("ACS (Steps)")
    table.add_column("Adoption %")
    table.add_column("Intensity (Requests)")
    table.add_column("Intensity (Adopters)")
    
    for cond in CONDITIONS:
        name = cond["name"]
        subset = [r for r in results if r["condition"] == name]
        if not subset: continue
        
        wins = [r for r in subset if r["success"]]
        acc = len(wins) / len(subset) * 100
        acs = statistics.mean([r["steps"] for r in wins]) if wins else 0.0
        
        # Adoption Logic
        adopters = [r for r in subset if r["entered_agent_view"] or r["agent_actions"] > 0]
        adoption = (len(adopters) / len(subset)) * 100
        
        # Intensity Logic
        intensity_all = statistics.mean([r["agent_actions"] for r in subset])
        intensity_adopted = statistics.mean([r["agent_actions"] for r in adopters]) if adopters else 0.0
        
        table.add_row(
            name, 
            f"{acc:.0f}%", 
            f"{acs:.1f}", 
            f"{adoption:.0f}%", 
            f"{intensity_all:.1f}", 
            f"{intensity_adopted:.1f}"
        )
            
    console.print(table)

    all_fps = set()
    for r in results:
        all_fps.update(r["fingerprints"])
    print(f"\nSystem Fingerprints observed: {all_fps}")

    # Save results to file
    from datetime import datetime
    from pathlib import Path

    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"run_{run_id}.json"

    output_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_iterations": MAX_ITERATIONS,
            "runs_per_task": RUNS_PER_TASK,
            "treatment_url": URL_TREATMENT,
            "baseline_url": URL_BASELINE
        },
        "results": results,
        "fingerprints": list(all_fps)
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())