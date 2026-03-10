"""
Graph builder — constructs the full LangGraph StateGraph.

Phase 1: START → planner → plan_reviewer → {pass→leader, fail→planner}
Phase 2: leader → Send(frontend_expert, backend_expert) → task_reviewer
         → {pass→leader, fail→expert, circuit_break→leader}
         → leader → ... → END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_team.schemas.state import GraphState
from agent_team.agents.planner import planner_node
from agent_team.agents.leader import leader_node
from agent_team.agents.experts import frontend_expert_node, backend_expert_node
from agent_team.agents.reviewer import reviewer_node


# ─────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────

def route_after_plan_review(state: GraphState) -> str:
    """Route after the plan reviewer: pass → leader, fail → planner."""
    actor = state.get("current_actor", "")
    if actor == "done":
        return END
    if actor == "leader":
        return "leader"
    return "planner"


def route_after_leader(state: GraphState) -> list[Send] | str:
    """Route after Leader: fan-out to experts via Send or go to END.

    This is the key mechanism for **async parallel execution**:
    returning multiple Send objects causes LangGraph to execute the
    destination nodes simultaneously in the same super-step.
    """
    actor = state.get("current_actor", "")
    if actor == "done":
        return END

    active = state.get("current_active_tasks", {})
    sends: list[Send] = []

    if "frontend" in active and active["frontend"]:
        sends.append(Send("frontend_expert", state))
    if "backend" in active and active["backend"]:
        sends.append(Send("backend_expert", state))

    if not sends:
        # Fallback: nothing to dispatch → end
        return END

    return sends


def route_after_task_review(state: GraphState) -> list[Send] | str:
    """Route after task reviewer: pass→leader, fail→expert, circuit_break→leader."""
    actors = state.get("current_actor", "")
    
    if isinstance(actors, list):
        # When multiple actors are listed, create Send objects for experts.
        # The reviewer already ensures "leader" is never mixed with experts
        # (to prevent race conditions), but we enforce it here too for safety.
        expert_sends = []
        for act in set(actors):
            if act in ["frontend_expert", "backend_expert"]:
                expert_sends.append(Send(act, state))
        if expert_sends:
            return expert_sends
        elif "leader" in actors:
            return "leader"
        return "leader"
    else:
        if actors == "leader":
            return "leader"
        if actors == "frontend_expert":
            return "frontend_expert"
        if actors == "backend_expert":
            return "backend_expert"
        return "leader"


# ─────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and return the compiled multi-agent StateGraph.

    Returns:
        A compiled LangGraph graph ready to be invoked.
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ──
    graph.add_node("planner", planner_node)
    graph.add_node("plan_reviewer", reviewer_node)
    graph.add_node("leader", leader_node)
    graph.add_node("frontend_expert", frontend_expert_node)
    graph.add_node("backend_expert", backend_expert_node)
    graph.add_node("task_reviewer", reviewer_node)

    # ── Phase 1: Planning ──
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "plan_reviewer")
    graph.add_conditional_edges(
        "plan_reviewer",
        route_after_plan_review,
        {"leader": "leader", "planner": "planner", "__end__": END},
    )

    # ── Phase 2: Execution ──
    # Leader fans out to experts via Send (parallel dispatch)
    graph.add_conditional_edges(
        "leader",
        route_after_leader,
        ["frontend_expert", "backend_expert", "__end__"],
    )

    # Experts submit to task reviewer
    graph.add_edge("frontend_expert", "task_reviewer")
    graph.add_edge("backend_expert", "task_reviewer")

    # Task reviewer routes back
    graph.add_conditional_edges(
        "task_reviewer",
        route_after_task_review,
        ["leader", "frontend_expert", "backend_expert"],
    )

    return graph.compile()
