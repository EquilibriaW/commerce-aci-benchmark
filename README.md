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
```

### Running the Benchmark

**Terminal 1 - Start all servers:**
```bash
# Baseline (port 3001)
cd baseline && npm run dev &

# Treatment 1 - Terminal UI (port 3000)
cd treatment && npm run dev &

# Treatment 2 - Documentation UI (port 3002)
cd treatment-docs && npm run dev &
```

**Terminal 2 - Run benchmark:**
```bash
python benchmark_computeruse.py
```

## Understanding the Results

The benchmark reports:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Task completion rate (%) |
| **Avg Steps** | Average steps to complete successful tasks |
| **Adoption %** | How often agent discovered/used agent routes |
| **Agent Actions** | API calls to `/agent/actions/*` endpoints |

Results are saved to `benchmark_results/` as JSON files. Debug screenshots are saved to `debug_screenshots/`.

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

The verifier receives the cart state and returns `True` if the task succeeded:

```python
# State structure passed to verifier
state = {
    "cart": {
        "items": [
            {
                "slug": "black-t-shirt",
                "variant": "M",
                "quantity": 1,
                "price_cents": 2000
            }
        ],
        "total_items": 1,
        "total_price_cents": 2000
    }
}
```

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

#### GET `/agent/state` - Returns cart state
```json
{
  "cart": {
    "items": [
      {"slug": "product-handle", "variant": "M", "quantity": 1, "price_cents": 2000}
    ],
    "total_items": 1,
    "total_price_cents": 2000
  }
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

    # Your custom site
    {
        "name": "My Custom Site",
        "target_url": "http://localhost:4000",      # Starting URL
        "api_url": "http://localhost:4000"          # Base URL for API calls
    },

    # Test discovery (agent starts at root, must find agent interface)
    {
        "name": "My Site (Discovery)",
        "target_url": "http://localhost:4000",
        "api_url": "http://localhost:4000"
    },

    # Test direct agent access
    {
        "name": "My Site (Agent Direct)",
        "target_url": "http://localhost:4000/agent",
        "api_url": "http://localhost:4000"
    },
]
```

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
├── requirements.txt               # Python dependencies
│
├── benchmark_results/             # JSON output files
│   └── computeruse_YYYYMMDD_HHMMSS.json
│
├── debug_screenshots/             # Debug screenshots per run
│   └── {condition}/{task}/run_{n}/
│       ├── step_01.png
│       ├── step_02.png
│       └── actions.log
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
