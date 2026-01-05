"""
Example Tasks for Commerce ACI Benchmark

Copy these task definitions into benchmark_computeruse.py to test different scenarios.
Each task has:
- id: Unique identifier (used in output files and screenshots)
- instruction: Natural language task given to the agent
- verifier: Lambda function that checks if task succeeded

The verifier receives the cart state:
{
    "cart": {
        "items": [
            {"slug": "product-handle", "variant": "M", "quantity": 1, "price_cents": 2000}
        ],
        "total_items": 1,
        "total_price_cents": 2000
    }
}
"""

# =============================================================================
# BASIC TASKS - Simple product addition
# =============================================================================

BASIC_TASKS = [
    # Add any product (simplest case)
    {
        "id": "add_any_product",
        "instruction": "Add something to my cart",
        "verifier": lambda s: len(s['cart']['items']) > 0
    },

    # Add specific product by name
    {
        "id": "add_cup",
        "instruction": "Add an Acme cup to my cart",
        "verifier": lambda s: any(
            i['slug'] == 'acme-cup' for i in s['cart']['items']
        )
    },

    # Add product using natural description
    {
        "id": "add_shirt_natural",
        "instruction": "I need a t-shirt",
        "verifier": lambda s: any(
            'shirt' in i['slug'].lower() for i in s['cart']['items']
        )
    },
]

# =============================================================================
# VARIANT TASKS - Size, color, or other options
# =============================================================================

VARIANT_TASKS = [
    # Specific size
    {
        "id": "add_small_shirt",
        "instruction": "Get me a small black T-shirt",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and i.get('variant') == 'S'
            for i in s['cart']['items']
        )
    },

    {
        "id": "add_medium_shirt",
        "instruction": "I want a medium-sized black T-shirt",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and i.get('variant') == 'M'
            for i in s['cart']['items']
        )
    },

    {
        "id": "add_large_shirt",
        "instruction": "Add a large black T-shirt to my cart",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and i.get('variant') == 'L'
            for i in s['cart']['items']
        )
    },
]

# =============================================================================
# QUANTITY TASKS - Multiple items
# =============================================================================

QUANTITY_TASKS = [
    # Specific quantity
    {
        "id": "add_two_cups",
        "instruction": "I need 2 Acme cups",
        "verifier": lambda s: any(
            i['slug'] == 'acme-cup' and i['quantity'] >= 2
            for i in s['cart']['items']
        )
    },

    {
        "id": "add_five_items",
        "instruction": "Add 5 black T-shirts to my cart",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and i['quantity'] >= 5
            for i in s['cart']['items']
        )
    },

    # Minimum quantity
    {
        "id": "add_multiple_cups",
        "instruction": "Get me several cups, at least 3",
        "verifier": lambda s: any(
            i['slug'] == 'acme-cup' and i['quantity'] >= 3
            for i in s['cart']['items']
        )
    },
]

# =============================================================================
# MULTI-PRODUCT TASKS - Cart with multiple products
# =============================================================================

MULTI_PRODUCT_TASKS = [
    # Two specific products
    {
        "id": "cup_and_shirt",
        "instruction": "Add a cup and a T-shirt to my cart",
        "verifier": lambda s: (
            any(i['slug'] == 'acme-cup' for i in s['cart']['items']) and
            any('shirt' in i['slug'].lower() for i in s['cart']['items'])
        )
    },

    # Outfit combination
    {
        "id": "full_outfit",
        "instruction": "I need a complete outfit: hoodie, T-shirt, and cap",
        "verifier": lambda s: (
            any(i['slug'] == 'hoodie' for i in s['cart']['items']) and
            any('shirt' in i['slug'].lower() for i in s['cart']['items']) and
            any('cap' in i['slug'].lower() for i in s['cart']['items'])
        )
    },

    # Multiple of same category
    {
        "id": "two_different_shirts",
        "instruction": "Add two different shirts to my cart",
        "verifier": lambda s: len([
            i for i in s['cart']['items'] if 'shirt' in i['slug'].lower()
        ]) >= 2
    },
]

# =============================================================================
# PRICE CONSTRAINT TASKS - Budget limitations
# =============================================================================

PRICE_TASKS = [
    # Under specific amount
    {
        "id": "budget_under_50",
        "instruction": "Add items to my cart, but keep the total under $50",
        "verifier": lambda s: (
            len(s['cart']['items']) > 0 and
            s['cart']['total_price_cents'] < 5000
        )
    },

    # Exact total
    {
        "id": "exact_total",
        "instruction": "Buy 2 cups and a hoodie (should be exactly $80)",
        "verifier": lambda s: s['cart']['total_price_cents'] == 8000
    },

    # Maximize within budget
    {
        "id": "maximize_budget",
        "instruction": "Spend as close to $100 as possible without going over",
        "verifier": lambda s: (
            8000 <= s['cart']['total_price_cents'] <= 10000
        )
    },
]

# =============================================================================
# COMPLEX TASKS - Combining multiple constraints
# =============================================================================

COMPLEX_TASKS = [
    # Variant + quantity
    {
        "id": "three_large_shirts",
        "instruction": "Get me 3 large black T-shirts",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' and
            i.get('variant') == 'L' and
            i['quantity'] >= 3
            for i in s['cart']['items']
        )
    },

    # Multi-product + quantity
    {
        "id": "party_supplies",
        "instruction": "I'm hosting a party. Get me 5 cups and 2 hoodies",
        "verifier": lambda s: (
            any(i['slug'] == 'acme-cup' and i['quantity'] >= 5
                for i in s['cart']['items']) and
            any(i['slug'] == 'hoodie' and i['quantity'] >= 2
                for i in s['cart']['items'])
        )
    },

    # Variant + price constraint
    {
        "id": "budget_outfit",
        "instruction": "Get me a large T-shirt and a cup, spending under $40",
        "verifier": lambda s: (
            any(i['slug'] == 'black-t-shirt' and i.get('variant') == 'L'
                for i in s['cart']['items']) and
            any(i['slug'] == 'acme-cup' for i in s['cart']['items']) and
            s['cart']['total_price_cents'] < 4000
        )
    },
]

# =============================================================================
# EDGE CASE TASKS - Testing robustness
# =============================================================================

EDGE_CASE_TASKS = [
    # Vague instruction
    {
        "id": "vague_request",
        "instruction": "Get me something nice",
        "verifier": lambda s: len(s['cart']['items']) > 0
    },

    # Typo in product name
    {
        "id": "typo_product",
        "instruction": "Add a blck t-shirt to cart",
        "verifier": lambda s: any(
            i['slug'] == 'black-t-shirt' for i in s['cart']['items']
        )
    },

    # Casual language
    {
        "id": "casual_request",
        "instruction": "yo can u add a hoodie",
        "verifier": lambda s: any(
            i['slug'] == 'hoodie' for i in s['cart']['items']
        )
    },

    # Negative constraint (don't add certain item)
    {
        "id": "not_this",
        "instruction": "Add a shirt to my cart, but not the black one",
        "verifier": lambda s: (
            any('shirt' in i['slug'].lower() for i in s['cart']['items']) and
            not any(i['slug'] == 'black-t-shirt' for i in s['cart']['items'])
        )
    },
]

# =============================================================================
# DISCOVERY TASKS - For testing agent UI discovery
# =============================================================================

DISCOVERY_TASKS = [
    # Find agent interface
    {
        "id": "find_agent_ui",
        "instruction": "Look for a machine-friendly interface and add a product",
        "verifier": lambda s: len(s['cart']['items']) > 0
    },

    # Use specific route
    {
        "id": "use_llms_txt",
        "instruction": "Check the llms.txt file and add the first product you find",
        "verifier": lambda s: len(s['cart']['items']) > 0
    },
]


# =============================================================================
# HOW TO USE
# =============================================================================
"""
To use these tasks in your benchmark:

1. Import the tasks you want:

   from example_tasks import BASIC_TASKS, VARIANT_TASKS

2. Or copy the task definitions directly into benchmark_computeruse.py:

   TASKS = [
       {
           "id": "add_cup",
           "instruction": "Add an Acme cup to my cart",
           "verifier": lambda s: any(
               i['slug'] == 'acme-cup' for i in s['cart']['items']
           )
       },
       # ... more tasks
   ]

3. Run the benchmark:

   python benchmark_computeruse.py
"""

# Combine all tasks for convenience
ALL_TASKS = (
    BASIC_TASKS +
    VARIANT_TASKS +
    QUANTITY_TASKS +
    MULTI_PRODUCT_TASKS +
    PRICE_TASKS +
    COMPLEX_TASKS +
    EDGE_CASE_TASKS +
    DISCOVERY_TASKS
)

if __name__ == "__main__":
    print(f"Total example tasks available: {len(ALL_TASKS)}")
    print("\nCategories:")
    print(f"  - Basic tasks: {len(BASIC_TASKS)}")
    print(f"  - Variant tasks: {len(VARIANT_TASKS)}")
    print(f"  - Quantity tasks: {len(QUANTITY_TASKS)}")
    print(f"  - Multi-product tasks: {len(MULTI_PRODUCT_TASKS)}")
    print(f"  - Price constraint tasks: {len(PRICE_TASKS)}")
    print(f"  - Complex tasks: {len(COMPLEX_TASKS)}")
    print(f"  - Edge case tasks: {len(EDGE_CASE_TASKS)}")
    print(f"  - Discovery tasks: {len(DISCOVERY_TASKS)}")
