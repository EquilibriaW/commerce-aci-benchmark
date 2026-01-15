#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline guidance fragments for the commerce agent."""

from __future__ import annotations


def baseline_guidance(config: dict | None = None) -> list[str]:
    _ = config
    return [
        "When unsure, ask a clarifying question instead of guessing.",
        "Before any irreversible action, explicitly confirm.",
        "Prefer minimal actions; avoid loops.",
    ]
