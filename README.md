# Commerce ACI Benchmark

A reproducible benchmark for evaluating AI Agents on e-commerce tasks using Claude's Computer Use API.

## Overview

This benchmark tests whether AI agents perform better on **Agent-Optimized UIs** vs **Standard Human UIs**. It uses Claude Sonnet 4.5 with computer use (vision + mouse/keyboard) to complete e-commerce tasks.

### Test Conditions

| Condition | Directory | Port | Description |
|-----------|-----------|------|-------------|
| **Baseline (Control)** | `baseline/` | 3001 | Standard e-commerce UI - no agent optimizations |
| **Treatment 1: Terminal UI** | `treatment/` | 3000 | Terminal-style agent interface at `/agent` |
| **Treatment 2: Documentation UI** | `treatment-docs/` | 3002 | ReadTheDocs/Sphinx-style API documentation UI |

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/EquilibriaW/commerce-aci-benchmark.git
cd commerce-aci-benchmark

# Install Python dependencies
pip install anthropic playwright httpx pydantic rich

# Install Playwright browser
playwright install chromium

# Install Node dependencies for each variant
cd baseline && npm install --legacy-peer-deps && cd ..
cd treatment && npm install --legacy-peer-deps && cd ..
cd treatment-docs && npm install --legacy-peer-deps && cd ..
```

### Environment Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export BENCHMARK_SECRET="sk-bench-123"  # Optional, this is the default

# Note: Welcome toast is disabled by default during benchmarking.
# To enable: export NEXT_PUBLIC_SHOW_WELCOME_TOAST="1"
```

### Running the Benchmark

**Terminal 1 - Start all servers:**

> **Recommended**: Use production builds for stable benchmarking. Development servers may have slower response times and hot-reload artifacts.

```bash
# Build all variants first (production mode)
cd baseline && pnpm build && pnpm start -p 3001 &
cd treatment && pnpm build && pnpm start -p 3000 &
cd treatment-docs && pnpm build && pnpm start -p 3002 &

# Alternative: Development mode (faster startup, less stable)
# cd baseline && npm run dev &
# cd treatment && npm run dev &
# cd treatment-docs && npm run dev &
```

**Terminal 2 - Run benchmark:**
```bash
python benchmark_computeruse.py
```

## Understanding the Results

### Output Tables

The benchmark produces 5 summary tables:

1. **Results by Condition** - Full breakdown per experimental condition
2. **Results by App** - Aggregated by app (baseline, treatment, treatment-docs)
3. **Results by Capability** - Compares advantage vs parity modes
4. **Results by Discoverability** - Compares navbar vs hidden agent link visibility
5. **Conditional Success** - Success rate split by whether agent adopted the UI

### Metrics Glossary

| Metric | Description | Scope |
|--------|-------------|-------|
| **N** | Number of trials | All |
| **Accuracy** | Task completion rate (%) | All |
| **Steps** | Average steps to complete task | Wins only |
| **Model** | Average LLM API calls | Wins only |
| **UI** | Average UI actions (click, drag, type, key, scroll) | Wins only |
| **Time** | Average wall-clock seconds | Wins only |
| **Adopt** | % of trials where agent navigated to `/agent` | Discovery runs only |
| **Adopt@Step** | Average step number when agent first visited `/agent` | Adopters only |
| **AgentAPI** | Average calls to `/agent/actions/*` endpoints | Wins only |

> **Note**: Efficiency metrics (Steps, Model, UI, Time, AgentAPI) are computed over successful trials only, so you're comparing apples-to-apples efficiency.

### Experimental Factors

The benchmark uses a factorized experimental design with these independent variables:

| Factor | Values | Description |
|--------|--------|-------------|
| **App** | `baseline`, `treatment`, `treatment-docs` | Which UI variant |
| **Start** | Root (`/`), Agent (`/agent`) | Entry point for the agent |
| **Discoverability** | `navbar`, `hidden` | Whether `/agent` link is visible in nav |
| **Capability** | `advantage`, `parity` | Whether agent UI has interactive actions or is read-only |

**Key comparisons:**
- `advantage` vs `parity`: Measures ease-of-use (can agent complete task faster with actions?)
- `navbar` vs `hidden`: Measures willingness-to-adopt (does agent find/use the UI?)
- Baseline vs Treatment: Overall lift from agent-optimized UI

### Output Files

Results are saved to `benchmark_results/computeruse_YYYYMMDD_HHMMSS.json`:

```json
{
  "run_id": "20250106_120000",
  "model": "claude-sonnet-4-5-20250929",
  "config": { ... },
  "factors": {
    "apps": ["baseline", "treatment", "treatment-docs"],
    "discoverability": ["navbar", "hidden"],
    "capability": ["advantage", "parity"]
  },
  "factor_aggregates": {
    "by_app": { ... },
    "by_capability": { ... },
    "by_discoverability": { ... }
  },
  "results": [ ... per-trial results ... ]
}
```

Debug screenshots are saved to `debug_screenshots/{condition}/{task}/run_{n}/`.

### New: Structured Trace Artifacts ("VCR for agents")

Each run now also writes a `trace.json` into the same debug folder. This trace:

- records the **full trajectory** (step-by-step screenshots, LLM tool calls, tool results)
- enables **deterministic replay** of UI actions
- enables **counterfactual testing** on a frozen observation sequence

This is intentionally lightweight: screenshots are saved as files and referenced by path
in the JSON so the trace doesn't become huge.

### Replay a Trace (Reproduce Failures)

Re-execute a run's recorded UI actions (no LLM call required):

```bash
python replay_trace.py --trace debug_screenshots/.../trace.json --mode reexecute
```

### Counterfactual "Shadow" Replay

Run a different model/prompt against the same recorded screenshots and compare
action agreement / divergence:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python replay_trace.py --trace debug_screenshots/.../trace.json --mode shadow \
  --model claude-sonnet-4-5-20250929 \
  --system-prompt-file my_system_prompt.txt
```

This writes a `counterfactual_*.json` report next to the trace.

---

## Packs (Evaluators, Fuzzers, Guidance, Advisors)

Packs are drop-in folders that add evaluators and fuzzing strategies without touching core code.
They are discovered under `./packs/*/pack.toml` and any additional directories listed in
`COMMERCE_PACK_PATHS` (os.pathsep-separated).

### pack.toml schema

```toml
[pack]
id = "my_pack"
name = "My Pack"
version = "0.1.0"
description = "..."

[[evaluators]]
id = "my_eval"
kind = "code" # or "llm"
entrypoint = "evaluators.py:my_eval"
severity = "warn" # or "error"
description = "Human-readable evaluator summary."
default_config = { max_steps = 18 }

[[fuzzers]]
id = "my_fuzzer"
kind = "code" # template | llm | code
entrypoint = "fuzzers.py:build_cases"
description = "Human-readable fuzzer summary."
default_config = { base_instruction = "Buy me a black T-shirt" }

[[guidance]]
id = "baseline_guidance"
kind = "code" # static | code | llm
entrypoint = "guidance.py:baseline_guidance"
description = "Guidance fragments appended to the system prompt."

[[advisors]]
id = "suggest_patch"
kind = "llm" # or "code"
entrypoint = "advisor.py:suggest_patch"
description = "Suggest guidance patches based on failing traces."
default_config = { model = "claude-3-5-haiku-20241022" }
```

Entry points use `relative_file.py:function` syntax and must live inside the pack folder.

### Add a pack

1. Create a folder under `packs/your_pack` with a `pack.toml`.
2. Implement `evaluators.py` and/or `fuzzers.py`.
3. Optionally add an external pack folder to `COMMERCE_PACK_PATHS`.

### CLI usage

```bash
python pack_cli.py list-packs
python pack_cli.py list-components --type evaluators
python pack_cli.py eval-trace --trace debug_screenshots/.../trace.json --eval-packs commerce_safety
python pack_cli.py eval-dir --dir debug_screenshots --eval-packs commerce_safety --json
python pack_cli.py fuzz-generate --fuzz-pack basic_flow_fuzz --fuzzer basic_strategies --turns 2,3 --out fuzz_cases.json
python pack_cli.py guidance-suggest --trace debug_screenshots/.../trace.json --advisor-pack advisor_guidance_patch --advisor suggest_patch --out guidance_patch.json
```

### Built-in packs

- `commerce_safety`: deterministic guardrails (max steps, loop detector, premature TASK_COMPLETE)
- `basic_flow_fuzz`: template fuzzing (intent shift, info overload, tool injection)
- `guidance_basics`: simple guidance fragments to append to the system prompt
- `advisor_guidance_patch`: optional LLM advisor that proposes guidance patches

### Integrations

- Evaluate traces during a benchmark run:
  ```bash
  python benchmark_computeruse.py --eval-packs commerce_safety
  ```
- Optional gate policy (for blocking on uncertain results):
  ```bash
  python benchmark_computeruse.py --eval-packs commerce_safety --gate-policy policy.toml
  ```
- Apply guidance packs:
  ```bash
  python benchmark_computeruse.py --guidance-packs guidance_basics
  ```
- Use a pack fuzzer with flow fuzzing:
  ```bash
  python flow_fuzz.py --fuzz-pack basic_flow_fuzz --fuzzer basic_strategies --turns 3,4,5
  ```
- Evaluate fuzz traces and apply guidance:
  ```bash
  python flow_fuzz.py --fuzz-pack basic_flow_fuzz --fuzzer basic_strategies --turns 3,4,5 \
    --eval-packs commerce_safety --guidance-packs guidance_basics
  ```

## New: Trace Trees & Counterfactual Branching

The benchmark now includes a **git-like trace branching system** for counterfactual analysis. This allows you to:

- Import any trace as the root of a tree
- Create branches at any step with different interventions
- Test "what if" scenarios (different prompts, models, or forced actions)
- Compare outcomes across branches

### Creating a Trace Tree

```python
from trace_tree import TraceTree

# Import an existing trace as root
tree = TraceTree.create_from_existing_trace(
    trace_path="debug_screenshots/Treatment_Root/custom_123/trace.json",
    trees_dir="debug_screenshots/trees",
    description="Testing checkout flow variations"
)

print(f"Created tree: {tree.tree_id}")
```

### Creating Branches with Interventions

```python
from trace_tree import TraceTree, Intervention, InterventionType
from branch_executor import BranchExecutionConfig, run_branch_sync

# Load existing tree
tree = TraceTree(tree_id="abc12345", trees_dir="debug_screenshots/trees")

# Create a branch with a prompt intervention at step 5
intervention = Intervention(
    type=InterventionType.PROMPT_INSERT,
    prompt_text="IMPORTANT: Use the /agent page for faster task completion."
)

config = BranchExecutionConfig(
    tree=tree,
    parent_trace_id=tree.get_root().trace_id,
    branch_point_step=5,
    intervention=intervention,
    label="Prompt nudge at step 5"
)

result = run_branch_sync(config)
print(f"Branch success: {result.success}, steps: {result.steps_count}")
```

### Intervention Types

| Type | Description | Parameters |
|------|-------------|------------|
| `PROMPT_INSERT` | Inject text into the user message at branch point | `prompt_text` |
| `MODEL_SWAP` | Switch to a different model from the branch point | `model` (sonnet/haiku) |
| `TOOL_OVERRIDE` | Force a specific action at the branch point | `forced_action` dict |

### Using the Control Panel

The **AgentOps Control Panel** provides a visual interface for trace trees:

```bash
streamlit run control_panel.py
```

Features:
- **Run Launcher**: Execute benchmark tasks or custom goals
- **Trace Viewer**: Step-by-step inspection with screenshots
- **Trace Trees**: Interactive DAG visualization, create branches, compare outcomes

---

## New: Server Manager

The `server_manager.py` module automatically manages Next.js dev servers:

```python
from server_manager import get_server_manager

manager = get_server_manager()

# Start a server (waits for health check)
if manager.start_server("treatment", wait_ready=True, timeout=60):
    print("Server ready!")
    url = manager.get_url("treatment")  # http://localhost:3000

# Check status
status = manager.get_status()
# {"treatment": {"port": 3000, "healthy": True, ...}, ...}

# Stop all servers on exit
manager.stop_all()
```

---

## New: Model Selection

Choose between Claude Sonnet 4.5 (default) or Claude Haiku 3.5:

```bash
# Use Sonnet (default, more capable)
python benchmark_computeruse.py --model sonnet

# Use Haiku (faster, cheaper)
python benchmark_computeruse.py --model haiku
```

---

## New: Custom Instructions (Ad-hoc Tasks)

Run one-off tasks without defining them in `TASKS`:

```bash
python benchmark_computeruse.py \
    --instruction "Find the cheapest product and add it to cart" \
    --app treatment \
    --discoverability navbar
```

The agent runs until it declares `TASK_COMPLETE` or hits max iterations.

---

## New: Adversarial Flow Fuzzing (Chaos Testing for Agents)

Most AgentOps stacks still rely on *static evaluation sets*. `flow_fuzz.py` adds a prototype
"chaos monkey" that injects perturbations (intent shift, info overload, tool injection)
at a specified turn, then measures robustness.

```bash
python flow_fuzz.py --app treatment --discoverability navbar --capability advantage \
  --turns 3,4,5 --runs-per-scenario 1
```

Outputs:
- `benchmark_results/fuzz_<timestamp>.json`
- `benchmark_results/fuzz_<timestamp>_heatmap.png`

---

# Customization Guide

## Adding Custom Tasks

Tasks are defined in `benchmark_computeruse.py`. Each task has three components:

```python
TASKS = [
    {
        "id": "t01_find_add_simple",           # Unique identifier
        "instruction": "Buy me a black T-shirt",  # Natural language task
        "verifier": lambda s: any(               # Success checker
            i['slug'] == 'black-t-shirt' and i['quantity'] >= 1
            for i in s['cart']['items']
        )
    },
]
```

### Task Structure

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique task identifier (used in output files) |
| `instruction` | string | Natural language instruction given to the agent |
| `verifier` | function | Lambda that checks if the task was completed correctly |

### Verifier Function

The verifier receives the state from `/agent/state` and returns `True` if the task succeeded:

```python
# State structure passed to verifier (from /agent/state endpoint)
state = {
    "cart": {
        "id": "cart_123",
        "items": [
            {
                "id": "variant_id",
                "slug": "black-t-shirt",
                "title": "Black T-Shirt",
                "variant": "M",
                "quantity": 1,
                "unit_price_cents": 2000,
                "line_total_cents": 2000
            }
        ],
        "total_items": 1,
        "total_price_cents": 2000,
        "currency": "USD"
    },
    "last_order": {  # Present after checkout completed
        "id": "order_123",
        "customer": {"name": "John", "email": "john@example.com"},
        "items": [...],
        "total_items": 1,
        "total_price_cents": 2000,
        "currency": "USD",
        "completed_at": "2025-01-06T12:00:00Z"
    }
}
```

> **Note**: Task verifiers check `last_order` (completed checkout) rather than `cart` to ensure agents complete the full checkout flow.

### Example Tasks

```python
# Simple: Add any quantity of a product
{
    "id": "add_cup",
    "instruction": "Add an Acme cup to my cart",
    "verifier": lambda s: any(
        i['slug'] == 'acme-cup' for i in s['cart']['items']
    )
}

# Variant selection: Specific size/color
{
    "id": "add_large_shirt",
    "instruction": "I need a large black T-shirt",
    "verifier": lambda s: any(
        i['slug'] == 'black-t-shirt' and i.get('variant') == 'L'
        for i in s['cart']['items']
    )
}

# Quantity: Multiple items
{
    "id": "add_three_cups",
    "instruction": "Get me 3 Acme cups",
    "verifier": lambda s: any(
        i['slug'] == 'acme-cup' and i['quantity'] >= 3
        for i in s['cart']['items']
    )
}

# Price constraint: Total under a limit
{
    "id": "budget_shopping",
    "instruction": "Buy 2 cups and a hoodie for under $90",
    "verifier": lambda s: s['cart']['total_price_cents'] <= 9000
}

# Multi-product: Specific combination
{
    "id": "outfit",
    "instruction": "Add a hoodie and a cap to my cart",
    "verifier": lambda s: (
        any(i['slug'] == 'hoodie' for i in s['cart']['items']) and
        any(i['slug'] == 'acme-cap' for i in s['cart']['items'])
    )
}
```

---

## Testing Your Own Site

You can test any website by adding a new condition to the benchmark.

### Step 1: Add Required Endpoints

Your site needs these API endpoints for the benchmark to verify results:

#### GET `/agent/state` - Returns cart state and completed order
```json
{
  "cart": {
    "id": "cart_123",
    "items": [
      {
        "id": "variant_id",
        "slug": "product-handle",
        "title": "Product Name",
        "variant": "M",
        "quantity": 1,
        "unit_price_cents": 2000,
        "line_total_cents": 2000
      }
    ],
    "total_items": 1,
    "total_price_cents": 2000,
    "currency": "USD"
  },
  "last_order": null,
  "timestamp": "2025-01-06T12:00:00Z"
}
```

#### POST `/agent/reset` - Clears cart for test isolation
Returns `200 OK` and sets a new session cookie.

Both endpoints should check the `X-Benchmark-Secret` header for security.

### Step 2: Add Your Condition

Add your site to the `CONDITIONS` list in `benchmark_computeruse.py`:

```python
CONDITIONS = [
    # ... existing conditions ...

    # Your custom site - full factorial design
    {
        "name": "MySite/Root/navbar/advantage",
        "app": "mysite",
        "target_url": "http://localhost:4000",
        "api_url": "http://localhost:4000",
        "discoverability": "navbar",   # Agent link visible
        "capability": "advantage",      # Interactive actions enabled
    },
    {
        "name": "MySite/Agent/navbar/advantage",
        "app": "mysite",
        "target_url": "http://localhost:4000/agent",  # Direct to agent UI
        "api_url": "http://localhost:4000",
        "discoverability": "navbar",
        "capability": "advantage",
    },
    {
        "name": "MySite/Root/hidden/advantage",
        "app": "mysite",
        "target_url": "http://localhost:4000",
        "api_url": "http://localhost:4000",
        "discoverability": "hidden",   # Agent link hidden
        "capability": "advantage",
    },
]
```

The `discoverability` and `capability` values are passed to `/agent/reset` as headers and set as cookies to control the UI behavior.

### Step 3: Run the Benchmark

```bash
python benchmark_computeruse.py
```

---

## Configuration Options

### Benchmark Settings

Edit these constants in `benchmark_computeruse.py`:

```python
# Display size for screenshots
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800

# Maximum steps before giving up on a task
MAX_ITERATIONS = 12

# Number of runs per task (for statistical significance)
RUNS_PER_TASK = 3

# Enable/disable debug screenshots
DEBUG_SCREENSHOTS = True
DEBUG_DIR = Path("debug_screenshots")
```

### Server URLs

Configure via environment variables or edit the defaults:

```python
URL_TREATMENT = os.getenv("URL_TREATMENT", "http://localhost:3000")
URL_TREATMENT_DOCS = os.getenv("URL_TREATMENT_DOCS", "http://localhost:3002")
URL_BASELINE = os.getenv("URL_BASELINE", "http://localhost:3001")
```

### Model Configuration

The benchmark uses Claude Sonnet 4.5 with computer use:

```python
model = "claude-sonnet-4-5-20250929"
betas = ["computer-use-2025-01-24"]
```

---

## Project Structure

```
commerce-aci-benchmark/
├── README.md                      # This file
├── benchmark_computeruse.py       # Main benchmark script (Claude Computer Use)
├── benchmark_science.py           # Alternative benchmark (OpenAI)
├── control_panel.py               # Streamlit UI for trace viewing & branching
├── replay_trace.py                # Trace replay and counterfactual shadow mode
├── flow_fuzz.py                   # Adversarial flow fuzzing
├── requirements.txt               # Python dependencies
│
├── trace_tree.py                  # Git-like trace branching system
├── branch_executor.py             # Hybrid replay-then-live execution
├── server_manager.py              # Auto-manage Next.js dev servers
├── tree_visualization.py          # DAG rendering for trace trees
│
├── benchmark_results/             # JSON output files
│   └── computeruse_YYYYMMDD_HHMMSS.json
│
├── debug_screenshots/             # Debug screenshots per run
│   ├── {condition}/{task}/run_{n}/
│   │   ├── step_01.png
│   │   ├── step_02.png
│   │   ├── trace.json
│   │   └── actions.log
│   └── trees/                     # Trace tree storage
│       └── {tree_id}/
│           ├── tree_index.json
│           ├── root/
│           └── branches/
│
├── baseline/                      # Control: Standard e-commerce UI
│   ├── app/
│   │   ├── agent/                # Required benchmark endpoints
│   │   │   ├── state/route.ts    # GET - cart state
│   │   │   └── reset/route.ts    # POST - reset cart
│   │   └── ...
│   └── lib/mock/                 # Mock product data
│
├── treatment/                     # Treatment 1: Terminal UI
│   ├── app/
│   │   ├── agent/
│   │   │   ├── page.tsx          # Terminal-style dashboard
│   │   │   ├── product/[slug]/   # Product pages with forms
│   │   │   ├── actions/add/      # Add to cart API
│   │   │   ├── state/            # Cart state
│   │   │   └── reset/            # Reset cart
│   │   └── llms.txt/             # Machine-readable catalog
│   └── middleware.ts             # Bot routing to /agent
│
└── treatment-docs/                # Treatment 2: Documentation UI
    ├── app/
    │   ├── agent/
    │   │   ├── page.tsx          # ReadTheDocs-style module index
    │   │   └── product/[slug]/   # Python class-style product docs
    │   └── ...
    └── ...
```

---

## Available Products (Mock Data)

The benchmark uses these mock products:

| Product | Slug | Price | Variants |
|---------|------|-------|----------|
| Black T-Shirt | `black-t-shirt` | $20.00 | S, M, L |
| Acme Cup | `acme-cup` | $15.00 | None |
| Hoodie | `hoodie` | $50.00 | None |
| Acme Cap | `acme-cap` | $25.00 | None |
| ... | ... | ... | ... |

See `lib/mock/data.ts` for the full product catalog.

---

## Debugging

### View Screenshots

Screenshots are saved to `debug_screenshots/{condition}/{task}/run_{n}/`:

```bash
# Open all screenshots for a specific run
open debug_screenshots/Treatment_1_Terminal_UI/t01_find_add_simple/run_01/
```

### View Action Logs

Each run saves an `actions.log` file:

```
Step 01: left_click - at (640, 300)
Step 02: scroll - delta_y=-200
Step 03: left_click - at (450, 520)
Step 04: TASK_COMPLETE - Agent declared task complete
```

### Common Issues

**Rate Limiting**: The benchmark waits 60s on rate limit errors and retries.

**Session Cookies**: If carts aren't persisting, check that `/agent/reset` returns a `Set-Cookie` header.

**Task Failures**: Check screenshots to see what the agent saw and did.

---

## Extending the Benchmark

### Adding a New UI Treatment

1. Copy `treatment/` to a new directory (e.g., `treatment-chat/`)
2. Modify the UI in `app/agent/`
3. Keep the same API endpoints (`/agent/state`, `/agent/reset`, `/agent/actions/add`)
4. Add your condition to `benchmark_computeruse.py`

### Changing the Agent Prompt

Edit `SYSTEM_PROMPT` in `benchmark_computeruse.py`:

```python
SYSTEM_PROMPT = """You are an AI agent operating a computer...

GUIDELINES:
1. **Analyze the Interface**: ...
2. **Prioritize Speed**: ...
...
"""
```

### Using a Different Model

Change the model in the `run_step` method:

```python
response = self.client.messages.create(
    model="claude-sonnet-4-5-20250929",  # Change this
    ...
)
```

---

## License

Based on [Vercel Commerce](https://github.com/vercel/commerce) - MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the benchmark to verify
5. Submit a pull request
