#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-test for pack discovery and evaluator runtime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pack_api.loader import PackRegistry
from pack_api.runtime import eval_results_to_dicts, gate_decision, run_evaluators_on_trace


def main() -> None:
    registry = PackRegistry.discover()
    packs = registry.list_packs()
    print(f"packs={len(packs)}")

    dummy_trace = {
        "schema_version": "trace.v2",
        "trace_version": "v2",
        "meta": {"task_instruction": "Buy me a large black T-shirt"},
        "steps": [
            {
                "step": 1,
                "observation": {"screenshot_path": "missing.png"},
                "assistant": {
                    "text": "Looking at the page.",
                    "tool_uses": [
                        {
                            "tool_use_id": "tool_1",
                            "name": "computer",
                            "input": {"action": "scroll", "scroll_direction": "down", "scroll_amount": 2},
                        }
                    ],
                },
                "tool_results": [
                    {
                        "tool_use_id": "tool_1",
                        "result_text": "Scrolled down",
                        "screenshot_path": "missing.png",
                    }
                ],
            }
        ],
    }

    results = run_evaluators_on_trace(dummy_trace, selected_packs=["commerce_safety"], registry=registry)
    for res in eval_results_to_dicts(results):
        print(res)
    print(f"gate_passed={gate_decision(results)}")


if __name__ == "__main__":
    main()
