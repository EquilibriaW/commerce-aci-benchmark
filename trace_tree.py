#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace Tree - Git-like tree structure for trace branching and counterfactuals.

This module manages traces as a tree structure where:
- Each trace has a unique trace_id
- Branches reference their parent via parent_trace_id
- The tree_index.json tracks all traces and their relationships

Usage:
    # Create a new tree from an existing trace
    tree = TraceTree.create_from_existing_trace(trace_path, trees_dir)

    # Load an existing tree
    tree = TraceTree(tree_id, trees_dir)

    # Create a branch
    new_id = tree.create_branch(parent_id, step=5, intervention=intervention)
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class InterventionType(Enum):
    """Types of interventions that can be applied at a branch point."""
    PROMPT_INSERT = "prompt_insert"
    MODEL_SWAP = "model_swap"
    TOOL_OVERRIDE = "tool_override"


@dataclass
class Intervention:
    """An intervention applied at a branch point."""
    type: InterventionType
    prompt_text: Optional[str] = None      # For PROMPT_INSERT
    model: Optional[str] = None            # For MODEL_SWAP
    forced_action: Optional[dict] = None   # For TOOL_OVERRIDE

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "prompt_text": self.prompt_text,
            "model": self.model,
            "forced_action": self.forced_action,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intervention":
        if d is None:
            return None
        return cls(
            type=InterventionType(d["type"]),
            prompt_text=d.get("prompt_text"),
            model=d.get("model"),
            forced_action=d.get("forced_action"),
        )


@dataclass
class TraceNode:
    """A node in the trace tree representing one trace."""
    trace_id: str
    parent_trace_id: Optional[str]
    branch_point_step: Optional[int]
    intervention: Optional[Intervention]
    path: Path
    label: str
    success: Optional[bool]
    steps_count: int
    created_at: str

    @property
    def is_root(self) -> bool:
        return self.parent_trace_id is None


@dataclass
class TreeIndex:
    """Index of all traces in a tree."""
    tree_id: str
    created_at: str
    root_trace_id: str
    description: str = ""
    traces: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tree_id": self.tree_id,
            "created_at": self.created_at,
            "root_trace_id": self.root_trace_id,
            "description": self.description,
            "traces": self.traces,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeIndex":
        return cls(
            tree_id=d["tree_id"],
            created_at=d["created_at"],
            root_trace_id=d["root_trace_id"],
            description=d.get("description", ""),
            traces=d.get("traces", {}),
        )


class TraceTree:
    """Manages a tree of traces with git-like branching."""

    SCHEMA_VERSION = "trace.v2"

    def __init__(self, tree_id: str, trees_dir: Path):
        """Load an existing tree.

        Args:
            tree_id: The UUID of the tree
            trees_dir: Base directory for all trees (e.g., debug_screenshots/trees)
        """
        self.tree_id = tree_id
        self.trees_dir = Path(trees_dir)
        self.base_dir = self.trees_dir / tree_id
        self.index_path = self.base_dir / "tree_index.json"

        if not self.index_path.exists():
            raise ValueError(f"Tree not found: {tree_id}")

        self._index = self._load_index()

    def _load_index(self) -> TreeIndex:
        """Load the tree index from disk."""
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return TreeIndex.from_dict(data)

    def _save_index(self) -> None:
        """Save the tree index to disk."""
        self.index_path.write_text(
            json.dumps(self._index.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    @classmethod
    def create_from_existing_trace(
        cls,
        trace_path: Path,
        trees_dir: Path,
        description: str = ""
    ) -> "TraceTree":
        """Import an existing trace.v1 as root of a new tree.

        Args:
            trace_path: Path to existing trace.json
            trees_dir: Base directory for trees
            description: Optional description for the tree

        Returns:
            A new TraceTree instance
        """
        trace_path = Path(trace_path)
        trees_dir = Path(trees_dir)

        # Load existing trace
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))

        # Generate IDs
        tree_id = str(uuid.uuid4())[:8]
        trace_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        # Create tree directory structure
        tree_dir = trees_dir / tree_id
        root_dir = tree_dir / "root"
        branches_dir = tree_dir / "branches"
        root_dir.mkdir(parents=True, exist_ok=True)
        branches_dir.mkdir(parents=True, exist_ok=True)

        # Copy screenshots to root directory
        source_dir = trace_path.parent
        for png in source_dir.glob("*.png"):
            shutil.copy2(png, root_dir / png.name)

        # Copy actions.log if exists
        actions_log = source_dir / "actions.log"
        if actions_log.exists():
            shutil.copy2(actions_log, root_dir / "actions.log")

        # Upgrade trace to v2 schema
        trace_data["schema_version"] = cls.SCHEMA_VERSION
        trace_data["trace_id"] = trace_id
        trace_data["parent_trace_id"] = None
        trace_data["branch_point_step"] = None
        trace_data["intervention"] = None

        # Update meta
        meta = trace_data.get("meta", {})
        meta["is_branched"] = False
        meta["branch_depth"] = 0
        meta["created_at"] = meta.get("run_timestamp", now)
        trace_data["meta"] = meta

        # Update screenshot paths to be relative to new location
        steps = trace_data.get("steps", [])
        for step in steps:
            obs = step.get("observation", {})
            if obs.get("screenshot_path"):
                old_path = Path(obs["screenshot_path"])
                obs["screenshot_path"] = str(root_dir / old_path.name)
            if obs.get("debug_screenshot_path"):
                old_path = Path(obs["debug_screenshot_path"])
                obs["debug_screenshot_path"] = str(root_dir / old_path.name)

            for tr in step.get("tool_results", []):
                if tr.get("screenshot_path"):
                    old_path = Path(tr["screenshot_path"])
                    tr["screenshot_path"] = str(root_dir / old_path.name)

        # Save upgraded trace
        trace_file = root_dir / "trace.json"
        trace_file.write_text(
            json.dumps(trace_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Determine success from trace
        success = None  # Could parse from original results if available

        # Create tree index
        index = TreeIndex(
            tree_id=tree_id,
            created_at=now,
            root_trace_id=trace_id,
            description=description or f"Imported from {trace_path.parent.name}",
            traces={
                trace_id: {
                    "parent_trace_id": None,
                    "branch_point_step": None,
                    "intervention_type": None,
                    "label": "Root",
                    "created_at": now,
                    "success": success,
                    "steps_count": len(steps),
                    "path": str(root_dir),
                }
            }
        )

        # Save index
        index_path = tree_dir / "tree_index.json"
        index_path.write_text(
            json.dumps(index.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return cls(tree_id, trees_dir)

    @classmethod
    def list_trees(cls, trees_dir: Path) -> list[dict]:
        """List all available trees.

        Returns:
            List of dicts with tree_id, description, created_at, trace_count
        """
        trees_dir = Path(trees_dir)
        if not trees_dir.exists():
            return []

        result = []
        for tree_path in trees_dir.iterdir():
            if not tree_path.is_dir():
                continue
            index_path = tree_path / "tree_index.json"
            if not index_path.exists():
                continue
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                result.append({
                    "tree_id": data.get("tree_id", tree_path.name),
                    "description": data.get("description", ""),
                    "created_at": data.get("created_at", ""),
                    "trace_count": len(data.get("traces", {})),
                })
            except Exception:
                continue

        return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_root(self) -> TraceNode:
        """Get the root trace node."""
        return self.get_node(self._index.root_trace_id)

    def get_node(self, trace_id: str) -> TraceNode:
        """Get a specific trace node by ID."""
        if trace_id not in self._index.traces:
            raise ValueError(f"Trace not found: {trace_id}")

        info = self._index.traces[trace_id]

        intervention = None
        if info.get("intervention_type"):
            # Load full intervention from trace file
            trace_data = self.load_trace(trace_id)
            intervention = Intervention.from_dict(trace_data.get("intervention"))

        return TraceNode(
            trace_id=trace_id,
            parent_trace_id=info.get("parent_trace_id"),
            branch_point_step=info.get("branch_point_step"),
            intervention=intervention,
            path=Path(info.get("path", "")),
            label=info.get("label", ""),
            success=info.get("success"),
            steps_count=info.get("steps_count", 0),
            created_at=info.get("created_at", ""),
        )

    def get_all_nodes(self) -> list[TraceNode]:
        """Get all trace nodes in the tree."""
        return [self.get_node(tid) for tid in self._index.traces]

    def get_children(self, trace_id: str) -> list[TraceNode]:
        """Get all direct children of a trace."""
        children = []
        for tid, info in self._index.traces.items():
            if info.get("parent_trace_id") == trace_id:
                children.append(self.get_node(tid))
        return children

    def get_ancestors(self, trace_id: str) -> list[TraceNode]:
        """Get path from root to this trace (excluding this trace).

        Returns ancestors in order from root to parent.
        """
        ancestors = []
        current_id = trace_id

        while True:
            info = self._index.traces.get(current_id)
            if not info:
                break
            parent_id = info.get("parent_trace_id")
            if not parent_id:
                break
            ancestors.insert(0, self.get_node(parent_id))
            current_id = parent_id

        return ancestors

    def create_branch(
        self,
        parent_trace_id: str,
        branch_point_step: int,
        intervention: Intervention,
        label: str = ""
    ) -> str:
        """Create a new branch placeholder.

        This creates the directory and index entry but doesn't populate the trace.
        Call save_branch_trace() after execution to complete.

        Returns:
            New trace_id
        """
        if parent_trace_id not in self._index.traces:
            raise ValueError(f"Parent trace not found: {parent_trace_id}")

        # Generate new trace ID
        trace_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        # Create branch directory
        branch_dir = self.base_dir / "branches" / trace_id
        branch_dir.mkdir(parents=True, exist_ok=True)

        # Add to index
        self._index.traces[trace_id] = {
            "parent_trace_id": parent_trace_id,
            "branch_point_step": branch_point_step,
            "intervention_type": intervention.type.value if intervention else None,
            "label": label or f"Branch at step {branch_point_step}",
            "created_at": now,
            "success": None,  # Will be set after execution
            "steps_count": 0,  # Will be set after execution
            "path": str(branch_dir),
        }

        self._save_index()
        return trace_id

    def save_branch_trace(
        self,
        trace_id: str,
        trace_data: dict,
        success: Optional[bool] = None
    ) -> None:
        """Save the completed branch trace.

        Args:
            trace_id: The trace ID (from create_branch)
            trace_data: Full trace data dict
            success: Whether the branch execution succeeded
        """
        if trace_id not in self._index.traces:
            raise ValueError(f"Trace not found: {trace_id}")

        info = self._index.traces[trace_id]
        branch_dir = Path(info["path"])

        # Save trace JSON
        trace_file = branch_dir / "trace.json"
        trace_file.write_text(
            json.dumps(trace_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Update index
        info["success"] = success
        info["steps_count"] = len(trace_data.get("steps", []))
        self._save_index()

    def load_trace(self, trace_id: str) -> dict:
        """Load full trace JSON for a node."""
        if trace_id not in self._index.traces:
            raise ValueError(f"Trace not found: {trace_id}")

        info = self._index.traces[trace_id]
        trace_path = Path(info["path"]) / "trace.json"

        if not trace_path.exists():
            raise ValueError(f"Trace file not found: {trace_path}")

        return json.loads(trace_path.read_text(encoding="utf-8"))

    def get_full_step_sequence(self, trace_id: str) -> list[dict]:
        """Get complete step sequence from root through ancestors to this trace.

        This combines steps from all ancestors and the current trace into
        a single sequence, useful for full replay.
        """
        all_steps = []

        # Get ancestors (root first)
        ancestors = self.get_ancestors(trace_id)

        for ancestor in ancestors:
            trace_data = self.load_trace(ancestor.trace_id)
            steps = trace_data.get("steps", [])

            # For ancestors, we only include steps up to the branch point
            # where the next trace branches off
            next_ancestor_idx = ancestors.index(ancestor) + 1
            if next_ancestor_idx < len(ancestors):
                next_ancestor = ancestors[next_ancestor_idx]
                # Include steps before the branch point
                branch_step = next_ancestor.branch_point_step
                if branch_step:
                    steps = steps[:branch_step - 1]

            all_steps.extend(steps)

        # Add steps from the target trace itself
        # For the target trace, we need to check if it's the direct child
        node = self.get_node(trace_id)
        if node.parent_trace_id and not ancestors:
            # Direct child of root - get root steps up to branch point
            root_data = self.load_trace(self._index.root_trace_id)
            root_steps = root_data.get("steps", [])
            if node.branch_point_step:
                root_steps = root_steps[:node.branch_point_step - 1]
            all_steps.extend(root_steps)

        # Now add the trace's own steps
        trace_data = self.load_trace(trace_id)
        all_steps.extend(trace_data.get("steps", []))

        return all_steps

    def get_steps_for_replay(self, trace_id: str, up_to_step: int) -> list[dict]:
        """Get steps needed to replay up to (but not including) a specific step.

        This is useful for branching: replay steps 1 to N-1, then intervene at N.

        Args:
            trace_id: The trace to replay within
            up_to_step: Stop before this step (1-indexed)

        Returns:
            List of step dicts to replay
        """
        full_sequence = self.get_full_step_sequence(trace_id)

        # up_to_step is 1-indexed, we want steps before it
        return full_sequence[:up_to_step - 1]

    def delete_branch(self, trace_id: str) -> None:
        """Delete a branch and all its descendants.

        Cannot delete the root trace.
        """
        if trace_id == self._index.root_trace_id:
            raise ValueError("Cannot delete root trace")

        if trace_id not in self._index.traces:
            raise ValueError(f"Trace not found: {trace_id}")

        # Find all descendants
        to_delete = [trace_id]
        for tid in list(self._index.traces.keys()):
            if self._is_descendant_of(tid, trace_id):
                to_delete.append(tid)

        # Delete directories and index entries
        for tid in to_delete:
            info = self._index.traces.get(tid)
            if info:
                branch_dir = Path(info["path"])
                if branch_dir.exists():
                    shutil.rmtree(branch_dir)
                del self._index.traces[tid]

        self._save_index()

    def _is_descendant_of(self, trace_id: str, ancestor_id: str) -> bool:
        """Check if trace_id is a descendant of ancestor_id."""
        current = trace_id
        while current:
            info = self._index.traces.get(current)
            if not info:
                return False
            parent = info.get("parent_trace_id")
            if parent == ancestor_id:
                return True
            current = parent
        return False
