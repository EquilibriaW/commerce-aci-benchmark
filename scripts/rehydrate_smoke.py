#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for trace rehydration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trace_rehydrate import build_conversation_from_trace


def main() -> None:
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "trace_fixture.json"
    trace = json.loads(fixture_path.read_text(encoding="utf-8"))
    messages = build_conversation_from_trace(trace, trace_dir=fixture_path.parent)
    print(f"messages={len(messages)}")
    if messages:
        print(f"first_role={messages[0].get('role')}")
        print(f"last_role={messages[-1].get('role')}")


if __name__ == "__main__":
    main()
