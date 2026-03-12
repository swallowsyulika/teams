"""
Graph builder — constructs the full LangGraph StateGraph.

Phase 1: START → planner → plan_reviewer → {pass→leader, fail→planner}
Phase 2: leader → Send(frontend_subgraph, backend_subgraph)
         Each subgraph internally loops:
             task_selector → expert → task_reviewer → task_selector → ...
         until its domain queue is empty → END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_team.schemas.state import GraphState, DomainState
from agent_team.agents.planner import planner_node
from agent_team.agents.leader import leader_node
from agent_team.agents.experts import task_selector_node, expert_node
from agent_team.agents.reviewer import plan_reviewer_node, task_reviewer_node


# ─────────────────────────────────────────────────
# Domain subgraph (shared by frontend & backend)
# ─────────────────────────────────────────────────

def _route_after_task_selector(state: DomainState) -> str:
    """Route after task_selector: if a task was found → expert, else → END."""
    if state.get("current_task_id", ""):
        return "expert"
    return END


def _route_after_task_review(state: DomainState) -> str:
    """Route after task_reviewer inside the subgraph.

    - If the current task is still ``in_progress`` (reviewer rejected,
      retries remain) → retry by going back to ``expert``.
    - Otherwise (task is ``completed`` or ``failed``) → go to
      ``task_selector`` to pick the next task.
    """
    task_id = state.get("current_task_id", "")
    task_list = state.get("task_list", [])

    for t in task_list:
        if t["id"] == task_id:
            if t["status"] == "in_progress":
                # Reviewer rejected but retries remain → retry
                return "expert"
            # completed or failed → pick next task
            return "task_selector"

    # Fallback: task not found → try to pick next
    return "task_selector"


def build_domain_subgraph() -> StateGraph:
    """Build and compile a domain subgraph.

    The subgraph loops: task_selector → expert → task_reviewer → ...
    until no pending tasks remain for this domain.

    Returns:
        A compiled subgraph that can be used as a node in the main graph.
    """
    sg = StateGraph(DomainState)

    sg.add_node("task_selector", task_selector_node)
    sg.add_node("expert", expert_node)
    sg.add_node("task_reviewer", task_reviewer_node)

    # START → task_selector
    sg.add_edge(START, "task_selector")

    # task_selector → expert (if task found) or END (queue empty)
    sg.add_conditional_edges(
        "task_selector",
        _route_after_task_selector,
        {"expert": "expert", "__end__": END},
    )

    # expert → task_reviewer
    sg.add_edge("expert", "task_reviewer")

    # task_reviewer → expert (retry) or task_selector (next task)
    sg.add_conditional_edges(
        "task_reviewer",
        _route_after_task_review,
        {"expert": "expert", "task_selector": "task_selector"},
    )

    return sg.compile()


# ─────────────────────────────────────────────────
# Main graph routing functions
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
    """Route after Leader: fan-out to domain subgraphs via Send or END.

    Each domain gets its own subgraph invocation with filtered tasks.
    The subgraphs run independently — a fast domain never blocks a
    slow one.
    """
    actor = state.get("current_actor", "")
    if actor == "done":
        return END

    task_list = state.get("task_list", [])
    code_base = state.get("code_base", {})
    system_design = state.get("system_design", "")

    sends: list[Send] = []

    # Check which domains have pending tasks
    fe_tasks = [t for t in task_list if t["domain"] == "frontend"]
    be_tasks = [t for t in task_list if t["domain"] == "backend"]

    has_fe_pending = any(t["status"] == "pending" for t in fe_tasks)
    has_be_pending = any(t["status"] == "pending" for t in be_tasks)

    if has_fe_pending:
        sends.append(Send("frontend_subgraph", {
            "domain": "frontend",
            "task_list": fe_tasks,
            "code_base": code_base,
            "system_design": system_design,
            "current_task_id": "",
            "review_feedback": "",
            "retry_count": 0,
        }))

    if has_be_pending:
        sends.append(Send("backend_subgraph", {
            "domain": "backend",
            "task_list": be_tasks,
            "code_base": code_base,
            "system_design": system_design,
            "current_task_id": "",
            "review_feedback": "",
            "retry_count": 0,
        }))

    if not sends:
        return END

    return sends


# ─────────────────────────────────────────────────
# Main graph construction
# ─────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and return the compiled multi-agent StateGraph.

    Returns:
        A compiled LangGraph graph ready to be invoked.
    """
    # Build the shared domain subgraph
    domain_subgraph = build_domain_subgraph()

    graph = StateGraph(GraphState)

    # ── Register nodes ──
    graph.add_node("planner", planner_node)
    graph.add_node("plan_reviewer", plan_reviewer_node)
    graph.add_node("leader", leader_node)
    graph.add_node("frontend_subgraph", domain_subgraph)
    graph.add_node("backend_subgraph", domain_subgraph)

    # ── Phase 1: Planning ──
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "plan_reviewer")
    graph.add_conditional_edges(
        "plan_reviewer",
        route_after_plan_review,
        {"leader": "leader", "planner": "planner", "__end__": END},
    )

    # ── Phase 2: Execution ──
    # Leader fans out to domain subgraphs via Send (parallel dispatch)
    graph.add_conditional_edges(
        "leader",
        route_after_leader,
        ["frontend_subgraph", "backend_subgraph", "__end__"],
    )

    return graph.compile()
