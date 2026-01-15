#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace schema validation helpers."""

from __future__ import annotations

from typing import Any


def validate_trace_v2(trace: dict[str, Any]) -> None:
    """Validate minimal trace.v2 ABI requirements.

    Raises:
        ValueError: if the trace is missing required fields or has the wrong version.
    """
    trace_version = trace.get("trace_version")
    schema_version = trace.get("schema_version")

    if trace_version is None:
        if schema_version == "trace.v2":
            trace_version = "v2"

    if trace_version != "v2":
        raise ValueError(
            f"Invalid trace_version: {trace_version!r}. Expected 'v2' (schema_version trace.v2)."
        )

    if "meta" not in trace or trace.get("meta") is None:
        raise ValueError("Invalid trace: missing meta")

    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Invalid trace: steps must be a list")

    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Invalid trace: step {idx} must be an object")
        obs = step.get("observation")
        if not isinstance(obs, dict):
            raise ValueError(f"Invalid trace: step {idx} missing observation")
        if "screenshot_path" not in obs:
            raise ValueError(f"Invalid trace: step {idx} missing observation.screenshot_path")
