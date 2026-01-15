#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack contracts for evaluators, fuzzers, guidance, and advisors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvaluatorSpec:
    id: str
    kind: str  # code | llm
    entrypoint: str
    default_config: dict[str, Any] = field(default_factory=dict)
    severity: str = "warn"
    description: str = ""


@dataclass
class FuzzerSpec:
    id: str
    kind: str  # template | llm | code
    entrypoint: str
    default_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class GuidanceSpec:
    id: str
    kind: str  # static | code | llm
    entrypoint: Optional[str] = None
    text_fragments: list[str] = field(default_factory=list)
    default_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class AdvisorSpec:
    id: str
    kind: str  # llm | code
    entrypoint: str
    default_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class PackManifest:
    id: str
    name: str
    version: str
    description: str = ""
    evaluators: list[EvaluatorSpec] = field(default_factory=list)
    fuzzers: list[FuzzerSpec] = field(default_factory=list)
    guidance: list[GuidanceSpec] = field(default_factory=list)
    advisors: list[AdvisorSpec] = field(default_factory=list)


@dataclass
class EvalResult:
    pack_id: str
    evaluator_id: str
    decision: str  # pass | fail | uncertain
    severity: str  # error | warn
    confidence: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GuidancePatch:
    patch_id: str
    base_guidance_id: str
    new_fragments: Optional[list[str]] = None
    diff: Optional[str] = None
    rationale: str = ""
    confidence: Optional[float] = None


@dataclass
class FuzzCase:
    scenario_id: str
    base_instruction: str
    injection_turn: int
    injection_message: str
    updated_goal: Optional[str] = None
    strategy: Optional[str] = None
