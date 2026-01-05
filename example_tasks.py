"""
Example Tasks for Commerce ACI Benchmark

Copy these task definitions into benchmark_computeruse.py to test different scenarios.

IMPORTANT: Tasks now require COMPLETED CHECKOUT (not just items in cart).
The agent must:
1. Add items to cart
2. Navigate to checkout
3. Fill in name and email
4. Submit the checkout form

This makes the comparison between human UI and agent UI fair.

Each task has:
- id: Unique identifier (used in output files and screenshots)
- instruction: Natural language task given to the agent
- verifier: Lambda function that checks if task succeeded

The verifier receives the state which includes:
{
    "cart": {...},  # Current cart (empty after checkout)
    "last_order": {  # Completed order (set after checkout)
        "id": "order_123",
        "customer": {"name": "John Doe", "email": "john@example.com"},
        "items": [
            {"slug": "product-handle", "variant": "M", "quantity": 1, "unit_price_cents": 2000}
        ],
        "total_items": 1,
        "total_price_cents": 2000,
        "completed_at": "2024-01-01T00:00:00.000Z"
    }
}
"""

# =============================================================================
# HELPER FUNCTIONS (copy these into benchmark_computeruse.py)
# =============================================================================

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


# =============================================================================
# BASIC TASKS - Simple product purchase
# =============================================================================

BASIC_TASKS = [
    # Buy any product (simplest case)
    {
        "id": "buy_any_product",
        "instruction": "Buy something from the store",
        "verifier": lambda s: (
            has_completed_order(s) and
            len(get_order_items(s)) > 0
        )
    },

    # Buy specific product by name
    {
        "id": "buy_cup",
        "instruction": "Buy an Acme cup",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'acme-cup' for i in get_order_items(s))
        )
    },

    # Buy product using natural description
    {
        "id": "buy_shirt_natural",
        "instruction": "I need to purchase a t-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any('shirt' in i['slug'].lower() for i in get_order_items(s))
        )
    },
]

# =============================================================================
# VARIANT TASKS - Size, color, or other options
# =============================================================================

VARIANT_TASKS = [
    # Specific size
    {
        "id": "buy_small_shirt",
        "instruction": "Buy a small black T-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i.get('variant') == 'S'
                for i in get_order_items(s))
        )
    },

    {
        "id": "buy_medium_shirt",
        "instruction": "Purchase a medium-sized black T-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i.get('variant') == 'M'
                for i in get_order_items(s))
        )
    },

    {
        "id": "buy_large_shirt",
        "instruction": "Buy a large black T-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i.get('variant') == 'L'
                for i in get_order_items(s))
        )
    },
]

# =============================================================================
# QUANTITY TASKS - Multiple items
# =============================================================================

QUANTITY_TASKS = [
    # Specific quantity
    {
        "id": "buy_two_cups",
        "instruction": "Buy 2 Acme cups",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'acme-cup' and i['quantity'] >= 2
                for i in get_order_items(s))
        )
    },

    {
        "id": "buy_five_shirts",
        "instruction": "Purchase 5 black T-shirts",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and i['quantity'] >= 5
                for i in get_order_items(s))
        )
    },

    # Minimum quantity
    {
        "id": "buy_multiple_cups",
        "instruction": "Buy at least 3 cups",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'acme-cup' and i['quantity'] >= 3
                for i in get_order_items(s))
        )
    },
]

# =============================================================================
# MULTI-PRODUCT TASKS - Order with multiple products
# =============================================================================

MULTI_PRODUCT_TASKS = [
    # Two specific products
    {
        "id": "buy_cup_and_shirt",
        "instruction": "Buy a cup and a T-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'acme-cup' for i in get_order_items(s)) and
            any('shirt' in i['slug'].lower() for i in get_order_items(s))
        )
    },

    # Outfit combination
    {
        "id": "buy_full_outfit",
        "instruction": "Buy a complete outfit: hoodie, T-shirt, and cap",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'hoodie' for i in get_order_items(s)) and
            any('shirt' in i['slug'].lower() for i in get_order_items(s)) and
            any('cap' in i['slug'].lower() for i in get_order_items(s))
        )
    },
]

# =============================================================================
# PRICE CONSTRAINT TASKS - Budget limitations
# =============================================================================

PRICE_TASKS = [
    # Under specific amount
    {
        "id": "budget_under_50",
        "instruction": "Buy some items, but keep the total under $50",
        "verifier": lambda s: (
            has_completed_order(s) and
            len(get_order_items(s)) > 0 and
            get_order_total(s) < 5000
        )
    },

    # Exact total
    {
        "id": "exact_total_80",
        "instruction": "Buy 2 cups and a hoodie (should total exactly $80)",
        "verifier": lambda s: (
            has_completed_order(s) and
            get_order_total(s) == 8000
        )
    },

    # Maximize within budget
    {
        "id": "maximize_budget_100",
        "instruction": "Spend as close to $100 as possible without going over",
        "verifier": lambda s: (
            has_completed_order(s) and
            8000 <= get_order_total(s) <= 10000
        )
    },
]

# =============================================================================
# COMPLEX TASKS - Combining multiple constraints
# =============================================================================

COMPLEX_TASKS = [
    # Variant + quantity
    {
        "id": "buy_three_large_shirts",
        "instruction": "Buy 3 large black T-shirts",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' and
                i.get('variant') == 'L' and
                i['quantity'] >= 3
                for i in get_order_items(s))
        )
    },

    # Multi-product + quantity
    {
        "id": "party_supplies",
        "instruction": "I'm hosting a party. Buy 5 cups and 2 hoodies",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'acme-cup' and i['quantity'] >= 5
                for i in get_order_items(s)) and
            any(i['slug'] == 'hoodie' and i['quantity'] >= 2
                for i in get_order_items(s))
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
        "verifier": lambda s: (
            has_completed_order(s) and
            len(get_order_items(s)) > 0
        )
    },

    # Typo in product name
    {
        "id": "typo_product",
        "instruction": "Buy a blck t-shirt",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'black-t-shirt' for i in get_order_items(s))
        )
    },

    # Casual language
    {
        "id": "casual_request",
        "instruction": "yo can u get me a hoodie",
        "verifier": lambda s: (
            has_completed_order(s) and
            any(i['slug'] == 'hoodie' for i in get_order_items(s))
        )
    },
]


# =============================================================================
# HOW TO USE
# =============================================================================
"""
To use these tasks in your benchmark:

1. Copy the helper functions and tasks into benchmark_computeruse.py:

   def get_order_items(state: dict) -> list:
       ...

   def has_completed_order(state: dict) -> bool:
       ...

   TASKS = [
       {
           "id": "buy_cup",
           "instruction": "Buy an Acme cup",
           "verifier": lambda s: (
               has_completed_order(s) and
               any(i['slug'] == 'acme-cup' for i in get_order_items(s))
           )
       },
       # ... more tasks
   ]

2. Run the benchmark:

   python benchmark_computeruse.py

3. The agent must now:
   - Add items to cart
   - Navigate to /checkout
   - Fill in name and email
   - Submit the form
   - Only then will the verifier pass
"""

# Combine all tasks for convenience
ALL_TASKS = (
    BASIC_TASKS +
    VARIANT_TASKS +
    QUANTITY_TASKS +
    MULTI_PRODUCT_TASKS +
    PRICE_TASKS +
    COMPLEX_TASKS +
    EDGE_CASE_TASKS
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
    print("\nNOTE: All tasks require completed checkout (not just cart add).")
