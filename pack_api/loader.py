#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack discovery and entrypoint loading."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Optional

import tomllib

from pack_api.contracts import (
    AdvisorSpec,
    EvaluatorSpec,
    FuzzerSpec,
    GuidanceSpec,
    PackManifest,
)


def _normalize_severity(value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"warning", "warn"}:
        return "warn"
    if value in {"error", "err"}:
        return "error"
    return value or "warn"


def _parse_evaluator(raw: dict[str, Any]) -> EvaluatorSpec:
    return EvaluatorSpec(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "code")),
        entrypoint=str(raw.get("entrypoint", "")),
        default_config=dict(raw.get("default_config") or {}),
        severity=_normalize_severity(str(raw.get("severity", "warn"))),
        description=str(raw.get("description", "")),
    )


def _parse_fuzzer(raw: dict[str, Any]) -> FuzzerSpec:
    return FuzzerSpec(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "code")),
        entrypoint=str(raw.get("entrypoint", "")),
        default_config=dict(raw.get("default_config") or {}),
        description=str(raw.get("description", "")),
    )


def _parse_guidance(raw: dict[str, Any]) -> GuidanceSpec:
    fragments = raw.get("text_fragments") or []
    if isinstance(fragments, str):
        fragments = [fragments]
    return GuidanceSpec(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "static")),
        entrypoint=raw.get("entrypoint"),
        text_fragments=[str(f) for f in (fragments or [])],
        default_config=dict(raw.get("default_config") or {}),
        description=str(raw.get("description", "")),
    )


def _parse_advisor(raw: dict[str, Any]) -> AdvisorSpec:
    return AdvisorSpec(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "llm")),
        entrypoint=str(raw.get("entrypoint", "")),
        default_config=dict(raw.get("default_config") or {}),
        description=str(raw.get("description", "")),
    )


def _discover_pack_dirs(base_dir: Path, extra_paths: list[Path]) -> list[Path]:
    pack_dirs: list[Path] = []
    if base_dir.exists():
        for pack_toml in base_dir.glob("*/pack.toml"):
            pack_dirs.append(pack_toml.parent)

    for extra in extra_paths:
        if extra.is_file() and extra.name == "pack.toml":
            pack_dirs.append(extra.parent)
            continue
        if extra.is_dir():
            pack_toml = extra / "pack.toml"
            if pack_toml.exists():
                pack_dirs.append(extra)
                continue
            for candidate in extra.glob("*/pack.toml"):
                pack_dirs.append(candidate.parent)

    deduped = []
    seen = set()
    for p in pack_dirs:
        resolved = p.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


class PackRegistry:
    """Loads and stores pack manifests with entrypoint loaders."""

    def __init__(self, pack_dirs: list[Path]) -> None:
        self._pack_dirs = pack_dirs
        self._manifests: dict[str, PackManifest] = {}
        self._pack_paths: dict[str, Path] = {}
        self._entrypoint_cache: dict[tuple[str, str, str], Callable[..., Any]] = {}
        self._load_packs()

    @classmethod
    def discover(cls, base_dir: Optional[Path] = None) -> "PackRegistry":
        base_dir = base_dir or (Path.cwd() / "packs")
        env_paths = os.environ.get("COMMERCE_PACK_PATHS", "")
        extra_paths = [Path(p) for p in env_paths.split(os.pathsep) if p]
        pack_dirs = _discover_pack_dirs(base_dir, extra_paths)
        return cls(pack_dirs)

    def _load_packs(self) -> None:
        for pack_dir in self._pack_dirs:
            pack_toml = pack_dir / "pack.toml"
            if not pack_toml.exists():
                continue

            raw = tomllib.loads(pack_toml.read_text(encoding="utf-8"))
            pack_info = raw.get("pack") or {}
            pack_id = str(pack_info.get("id", "")).strip()
            if not pack_id:
                raise ValueError(f"Pack missing id in {pack_toml}")
            if pack_id in self._manifests:
                raise ValueError(f"Duplicate pack id: {pack_id}")

            evaluators_raw = raw.get("evaluators") or []
            fuzzers_raw = raw.get("fuzzers") or []
            guidance_raw = raw.get("guidance") or []
            advisors_raw = raw.get("advisors") or []

            evaluators = [_parse_evaluator(e) for e in evaluators_raw]
            fuzzers = [_parse_fuzzer(f) for f in fuzzers_raw]
            guidance = [_parse_guidance(g) for g in guidance_raw]
            advisors = [_parse_advisor(a) for a in advisors_raw]

            eval_ids = [e.id for e in evaluators if e.id]
            fuzz_ids = [f.id for f in fuzzers if f.id]
            guidance_ids = [g.id for g in guidance if g.id]
            advisor_ids = [a.id for a in advisors if a.id]
            if len(eval_ids) != len(set(eval_ids)):
                raise ValueError(f"Duplicate evaluator id in pack {pack_id}")
            if len(fuzz_ids) != len(set(fuzz_ids)):
                raise ValueError(f"Duplicate fuzzer id in pack {pack_id}")
            if len(guidance_ids) != len(set(guidance_ids)):
                raise ValueError(f"Duplicate guidance id in pack {pack_id}")
            if len(advisor_ids) != len(set(advisor_ids)):
                raise ValueError(f"Duplicate advisor id in pack {pack_id}")

            manifest = PackManifest(
                id=pack_id,
                name=str(pack_info.get("name", pack_id)),
                version=str(pack_info.get("version", "0.0.0")),
                description=str(pack_info.get("description", "")),
                evaluators=evaluators,
                fuzzers=fuzzers,
                guidance=guidance,
                advisors=advisors,
            )
            self._manifests[pack_id] = manifest
            self._pack_paths[pack_id] = pack_dir

    def list_packs(self) -> list[PackManifest]:
        return [self._manifests[k] for k in sorted(self._manifests)]

    def get_manifest(self, pack_id: str) -> PackManifest:
        if pack_id not in self._manifests:
            raise KeyError(f"Pack not found: {pack_id}")
        return self._manifests[pack_id]

    def get_pack_dir(self, pack_id: str) -> Path:
        if pack_id not in self._pack_paths:
            raise KeyError(f"Pack not found: {pack_id}")
        return self._pack_paths[pack_id]

    def get_evaluator_spec(self, pack_id: str, evaluator_id: str) -> EvaluatorSpec:
        manifest = self.get_manifest(pack_id)
        for spec in manifest.evaluators:
            if spec.id == evaluator_id:
                return spec
        raise KeyError(f"Evaluator not found: {pack_id}:{evaluator_id}")

    def get_fuzzer_spec(self, pack_id: str, fuzzer_id: str) -> FuzzerSpec:
        manifest = self.get_manifest(pack_id)
        for spec in manifest.fuzzers:
            if spec.id == fuzzer_id:
                return spec
        raise KeyError(f"Fuzzer not found: {pack_id}:{fuzzer_id}")

    def get_guidance_spec(self, pack_id: str, guidance_id: str) -> GuidanceSpec:
        manifest = self.get_manifest(pack_id)
        for spec in manifest.guidance:
            if spec.id == guidance_id:
                return spec
        raise KeyError(f"Guidance not found: {pack_id}:{guidance_id}")

    def get_advisor_spec(self, pack_id: str, advisor_id: str) -> AdvisorSpec:
        manifest = self.get_manifest(pack_id)
        for spec in manifest.advisors:
            if spec.id == advisor_id:
                return spec
        raise KeyError(f"Advisor not found: {pack_id}:{advisor_id}")

    def load_entrypoint(self, pack_id: str, entrypoint: str) -> Callable[..., Any]:
        pack_dir = self.get_pack_dir(pack_id)
        if ":" not in entrypoint:
            raise ValueError(f"Invalid entrypoint '{entrypoint}', expected file.py:function")
        file_part, func_name = entrypoint.split(":", 1)
        file_path = Path(file_part)
        if not file_path.is_absolute():
            file_path = pack_dir / file_part
        file_path = file_path.resolve()

        if pack_dir.resolve() not in file_path.parents and file_path != pack_dir.resolve():
            raise ValueError(f"Entrypoint path escapes pack dir: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Entrypoint file not found: {file_path}")

        cache_key = (pack_id, str(file_path), func_name)
        if cache_key in self._entrypoint_cache:
            return self._entrypoint_cache[cache_key]

        digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:8]
        module_name = f"pack_{pack_id}_{file_path.stem}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, func_name):
            raise AttributeError(f"Entrypoint not found: {entrypoint}")
        func = getattr(module, func_name)
        self._entrypoint_cache[cache_key] = func
        return func

    def load_evaluator(self, pack_id: str, evaluator_id: str) -> Callable[..., Any]:
        spec = self.get_evaluator_spec(pack_id, evaluator_id)
        return self.load_entrypoint(pack_id, spec.entrypoint)

    def load_fuzzer(self, pack_id: str, fuzzer_id: str) -> Callable[..., Any]:
        spec = self.get_fuzzer_spec(pack_id, fuzzer_id)
        return self.load_entrypoint(pack_id, spec.entrypoint)

    def load_guidance(self, pack_id: str, guidance_id: str) -> Callable[..., Any]:
        spec = self.get_guidance_spec(pack_id, guidance_id)
        if not spec.entrypoint:
            raise ValueError(f"Guidance spec missing entrypoint: {pack_id}:{guidance_id}")
        return self.load_entrypoint(pack_id, spec.entrypoint)

    def load_advisor(self, pack_id: str, advisor_id: str) -> Callable[..., Any]:
        spec = self.get_advisor_spec(pack_id, advisor_id)
        return self.load_entrypoint(pack_id, spec.entrypoint)
