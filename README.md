# Commerce ACI Benchmark

A reproducible benchmark for evaluating AI Agents on e-commerce tasks, built on [Vercel Commerce](https://github.com/vercel/commerce).

## Overview

This repository contains two variants of an e-commerce store for A/B testing AI agent performance:

| Variant | Directory | Port | Description |
|---------|-----------|------|-------------|
| **Treatment** | `treatment/` | 3000 | Full ACI (Agent-Computer Interface) layer with agent-friendly routes |
| **Baseline** | `baseline/` | 3001 | Standard e-commerce UI, no agent optimizations |

## What is ACI (Agent-Computer Interface)?

ACI is a "Shadow DOM" approach for AI agents - providing machine-readable endpoints alongside the human UI:

- **`/llms.txt`** - Machine-readable product catalog in markdown
- **`/agent`** - Simplified HTML dashboard (no JavaScript/Tailwind)
- **`/agent/product/[slug]`** - Native HTML forms for add-to-cart
- **`/agent/actions/add`** - JSON API for cart operations
- **`/agent/state`** - Current cart state (secured)
- **`/agent/reset`** - Reset cart for test isolation (secured)

## Quick Start

### Prerequisites

- Node.js 18+
- npm

### Installation

```bash
# Clone the repository
git clone https://github.com/EquilibriaW/commerce-aci-benchmark.git
cd commerce-aci-benchmark

# Install treatment dependencies
cd treatment
npm install --legacy-peer-deps

# Install baseline dependencies
cd ../baseline
npm install --legacy-peer-deps
```

### Running the Servers

```bash
# Terminal 1 - Treatment (port 3000)
cd treatment
npm run dev

# Terminal 2 - Baseline (port 3001)
cd baseline
npm run dev -- -p 3001
```

## Architecture

### Mock Data Provider

Both variants use a mock data provider (`lib/mock/`) instead of Shopify, enabling:
- No API keys required
- Consistent test data across runs
- File-based cart persistence (`.cart-storage/`)

### Products

20 mock products across categories: outerwear, accessories, footwear, basics.

### Middleware (Treatment Only)

The treatment variant includes middleware that routes bot user-agents to `/agent`:

```typescript
// Detected user agents: GPTBot, BenchmarkAgent, Claude, Anthropic, OpenAI
const isBot = /GPTBot|BenchmarkAgent|Claude|Anthropic|OpenAI/i.test(userAgent);
```

Bypass with `?mode=human` query parameter.

## Benchmark API

### Secured Endpoints

These endpoints require the `X-Benchmark-Secret` header:

```bash
# Default secret (configure via BENCHMARK_SECRET env var)
X-Benchmark-Secret: sk-bench-123
```

#### Get Cart State
```bash
curl -H "X-Benchmark-Secret: sk-bench-123" \
  http://localhost:3000/agent/state
```

Response:
```json
{
  "cart": {
    "id": "cart_123",
    "items": [...],
    "total_items": 2,
    "total_price": 299.00,
    "currency": "USD"
  },
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

#### Reset State
```bash
curl -X POST \
  -H "X-Benchmark-Secret: sk-bench-123" \
  http://localhost:3000/agent/reset
```

### Agent Actions

#### Add to Cart
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"variantId": "var-1-m", "quantity": 1}' \
  http://localhost:3000/agent/actions/add
```

## Benchmark Workflow

```python
# Example Python benchmark harness
import requests

TREATMENT_URL = "http://localhost:3000"
BASELINE_URL = "http://localhost:3001"
SECRET = "sk-bench-123"

def reset_state(base_url):
    requests.post(
        f"{base_url}/agent/reset",
        headers={"X-Benchmark-Secret": SECRET}
    )

def get_state(base_url):
    return requests.get(
        f"{base_url}/agent/state",
        headers={"X-Benchmark-Secret": SECRET}
    ).json()

def run_agent_task(base_url, task):
    reset_state(base_url)
    # ... run your agent ...
    return get_state(base_url)

# Compare treatment vs baseline
treatment_result = run_agent_task(TREATMENT_URL, "Add a leather jacket to cart")
baseline_result = run_agent_task(BASELINE_URL, "Add a leather jacket to cart")
```

## Directory Structure

```
commerce-aci-benchmark/
├── README.md
├── treatment/                 # ACI-enabled variant
│   ├── app/
│   │   ├── agent/            # Agent-specific routes
│   │   │   ├── page.tsx      # Agent dashboard
│   │   │   ├── product/      # Agent product pages
│   │   │   ├── actions/      # JSON APIs
│   │   │   ├── state/        # Cart state endpoint
│   │   │   └── reset/        # Reset endpoint
│   │   └── llms.txt/         # Machine-readable catalog
│   ├── lib/mock/             # Mock data provider
│   ├── middleware.ts         # Bot routing
│   └── ...
└── baseline/                  # Control variant (no ACI)
    ├── app/                   # Standard routes only
    ├── lib/mock/             # Same mock provider
    └── ...
```

## Configuration

### Environment Variables

Create `.env.local` in each variant:

```bash
# Optional: Custom benchmark secret
BENCHMARK_SECRET=your-secret-here

# Store name (displayed in UI)
COMPANY_NAME=Agent Store
```

## Key Differences: Treatment vs Baseline

| Feature | Treatment | Baseline |
|---------|-----------|----------|
| `/agent/*` routes | Yes | No |
| `/llms.txt` | Yes | No |
| Bot middleware | Yes | No |
| `data-agent-id` attributes | Yes | No |
| Mock provider | Yes | Yes |
| Human UI | Yes | Yes |
| Cart persistence | Yes | Yes |

## Development

### Adding New Products

Edit `lib/mock/data.ts` in both variants to maintain consistency.

### Customizing Agent Routes

Agent routes are in `treatment/app/agent/`. They use:
- Pure HTML (no Tailwind classes)
- Native form submissions
- JSON responses for actions

## License

Based on [Vercel Commerce](https://github.com/vercel/commerce) - MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes to both variants if applicable
4. Submit a pull request

## Acknowledgments

- [Vercel Commerce](https://github.com/vercel/commerce) - Base e-commerce template
- Built for AI agent research and benchmarking
