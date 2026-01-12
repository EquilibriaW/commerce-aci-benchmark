#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tree Visualization - DAG rendering for trace trees using streamlit-agraph.

This module provides visualization components for trace trees in the Streamlit UI.

Usage:
    from tree_visualization import render_trace_tree, render_tree_fallback

    # In Streamlit app:
    selected_id = render_trace_tree(tree)
    if selected_id:
        st.write(f"Selected: {selected_id}")
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from trace_tree import TraceTree, TraceNode


# Try to import streamlit-agraph, fall back gracefully if not installed
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False


def render_trace_tree(tree: TraceTree) -> Optional[str]:
    """Render trace tree as an interactive DAG.

    Args:
        tree: TraceTree instance to visualize

    Returns:
        Selected node's trace_id, or None if nothing selected
    """
    if not AGRAPH_AVAILABLE:
        return render_tree_fallback(tree)

    nodes = []
    edges = []

    # Build graph from tree
    all_nodes = tree.get_all_nodes()

    for node in all_nodes:
        # Determine node color
        if node.is_root:
            color = "#4169E1"  # Royal blue for root
        elif node.intervention:
            color = "#FFA500"  # Orange for intervention nodes
        elif node.success is True:
            color = "#90EE90"  # Light green for success
        elif node.success is False:
            color = "#FFB6C1"  # Light pink for failure
        else:
            color = "#D3D3D3"  # Light gray for unknown/in-progress

        # Build label
        label = node.label or node.trace_id[:8]
        if node.intervention:
            intervention_short = node.intervention.type.value.replace("_", " ").title()
            label += f"\n({intervention_short})"
        if node.steps_count:
            label += f"\n[{node.steps_count} steps]"

        nodes.append(Node(
            id=node.trace_id,
            label=label,
            size=30 if node.is_root else 25,
            color=color,
            font={"size": 12}
        ))

        # Add edge to parent
        if node.parent_trace_id:
            edge_label = f"step {node.branch_point_step}" if node.branch_point_step else ""
            edges.append(Edge(
                source=node.parent_trace_id,
                target=node.trace_id,
                label=edge_label,
                color="#888888"
            ))

    # Configure the graph
    config = Config(
        width=800,
        height=400,
        directed=True,
        hierarchical=True,
        physics=False,  # Disable physics for cleaner layout
        highlightColor="#F7A7A6",
        nodeHighlightBehavior=True,
        collapsible=False,
    )

    # Render and return selection
    return agraph(nodes=nodes, edges=edges, config=config)


def render_tree_fallback(tree: TraceTree) -> Optional[str]:
    """Fallback tree rendering when streamlit-agraph is not available.

    Uses a simple indented list with selectbox.
    """
    st.warning("Install streamlit-agraph for interactive tree visualization: `pip install streamlit-agraph`")

    # Build tree structure for display
    def build_tree_lines(node: TraceNode, depth: int = 0) -> list[tuple[str, str]]:
        """Build display lines with indentation."""
        lines = []

        # Format this node
        prefix = "  " * depth + ("└─ " if depth > 0 else "")
        status = "✓" if node.success else ("✗" if node.success is False else "○")

        intervention_str = ""
        if node.intervention:
            intervention_str = f" [{node.intervention.type.value}]"

        line = f"{prefix}{status} {node.label or node.trace_id[:8]}{intervention_str} ({node.steps_count} steps)"
        lines.append((node.trace_id, line))

        # Recurse for children
        children = tree.get_children(node.trace_id)
        for child in children:
            lines.extend(build_tree_lines(child, depth + 1))

        return lines

    root = tree.get_root()
    lines = build_tree_lines(root)

    # Create selectbox options
    options = {tid: line for tid, line in lines}

    selected = st.selectbox(
        "Select trace",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        key="tree_select"
    )

    return selected


def render_node_details(tree: TraceTree, trace_id: str) -> None:
    """Render details panel for a selected trace node.

    Args:
        tree: TraceTree instance
        trace_id: ID of the trace to display
    """
    node = tree.get_node(trace_id)

    st.markdown(f"### {node.label or 'Trace ' + node.trace_id[:8]}")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Steps", node.steps_count)
    with col2:
        success_str = "Yes" if node.success else ("No" if node.success is False else "N/A")
        st.metric("Success", success_str)
    with col3:
        depth = len(tree.get_ancestors(trace_id))
        st.metric("Depth", depth)
    with col4:
        children = tree.get_children(trace_id)
        st.metric("Children", len(children))

    # Intervention info
    if node.intervention:
        st.markdown("#### Intervention")
        st.info(f"**Type:** {node.intervention.type.value}")
        if node.intervention.prompt_text:
            st.text_area("Prompt", node.intervention.prompt_text, disabled=True, height=100)
        if node.intervention.model:
            st.text(f"Model: {node.intervention.model}")
        if node.intervention.forced_action:
            st.json(node.intervention.forced_action)

    # Metadata
    with st.expander("Trace Metadata"):
        try:
            trace_data = tree.load_trace(trace_id)
            meta = trace_data.get("meta", {})
            st.json(meta)
        except Exception as e:
            st.error(f"Failed to load trace: {e}")

    # Path info
    st.caption(f"Path: {node.path}")
    st.caption(f"Created: {node.created_at}")


def render_step_preview(tree: TraceTree, trace_id: str, step_num: int) -> None:
    """Render a preview of a specific step.

    Args:
        tree: TraceTree instance
        trace_id: Trace ID
        step_num: Step number (1-indexed)
    """
    try:
        trace_data = tree.load_trace(trace_id)
        steps = trace_data.get("steps", [])

        if step_num < 1 or step_num > len(steps):
            st.error(f"Step {step_num} not found (trace has {len(steps)} steps)")
            return

        step = steps[step_num - 1]

        # Show screenshot
        obs = step.get("observation", {})
        screenshot_path = obs.get("screenshot_path")
        if screenshot_path:
            try:
                st.image(screenshot_path, caption=f"State at step {step_num}")
            except Exception:
                st.warning(f"Could not load screenshot: {screenshot_path}")

        # Show action taken
        assistant = step.get("assistant", {})
        tool_uses = assistant.get("tool_uses", [])
        if tool_uses:
            st.markdown("**Action taken:**")
            for tu in tool_uses:
                inp = tu.get("input", {})
                action = inp.get("action", "unknown")
                if action == "left_click":
                    coord = inp.get("coordinate", [0, 0])
                    st.code(f"click at ({coord[0]}, {coord[1]})")
                elif action == "type":
                    text = inp.get("text", "")
                    st.code(f"type: {text[:50]}{'...' if len(text) > 50 else ''}")
                elif action == "scroll":
                    direction = inp.get("scroll_direction", "down")
                    st.code(f"scroll {direction}")
                elif action == "key":
                    key = inp.get("key", "")
                    st.code(f"key: {key}")
                else:
                    st.code(f"{action}")

        # Show assistant reasoning
        if assistant.get("text"):
            with st.expander("Assistant reasoning"):
                st.text(assistant.get("text", "")[:500])

    except Exception as e:
        st.error(f"Failed to load step: {e}")
