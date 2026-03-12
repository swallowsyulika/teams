"""
Tests for graph construction — verifies the StateGraph compiles and
nodes/edges are wired correctly, without requiring an LLM API key.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_team.graph.builder import (
    build_graph,
    build_domain_subgraph,
    route_after_plan_review,
    route_after_leader,
    _route_after_task_selector,
    _route_after_task_review,
)
from langgraph.graph import END
from langgraph.types import Send


# ── Graph compilation ──────────────────────────

class TestGraphCompilation:
    def test_graph_compiles_successfully(self):
        """The main graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """All expected node names should be present in the compiled graph."""
        graph = build_graph()
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}
        expected = {"planner", "plan_reviewer", "leader",
                    "frontend_subgraph", "backend_subgraph"}
        assert expected.issubset(node_ids), f"Missing nodes: {expected - node_ids}"

    def test_domain_subgraph_compiles_successfully(self):
        """The domain subgraph should compile without errors."""
        sg = build_domain_subgraph()
        assert sg is not None


# ── Routing: Plan Review ──────────────────────

class TestRouteAfterPlanReview:
    def test_pass_routes_to_leader(self):
        state = {"current_actor": "leader"}
        assert route_after_plan_review(state) == "leader"

    def test_fail_routes_to_planner(self):
        state = {"current_actor": "planner"}
        assert route_after_plan_review(state) == "planner"

    def test_done_routes_to_end(self):
        state = {"current_actor": "done"}
        assert route_after_plan_review(state) == END

    def test_empty_actor_routes_to_planner(self):
        state = {"current_actor": ""}
        assert route_after_plan_review(state) == "planner"


# ── Routing: Leader Dispatch ──────────────────

class TestRouteAfterLeader:
    def test_done_routes_to_end(self):
        state = {"current_actor": "done", "task_list": []}
        result = route_after_leader(state)
        assert result == END

    def test_dispatches_both_subgraphs_in_parallel(self):
        state = {
            "current_actor": "dispatching",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "pending"},
                {"id": "be_1", "description": "Build API", "domain": "backend", "status": "pending"},
            ],
            "code_base": {},
            "system_design": "React + FastAPI",
        }
        result = route_after_leader(state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)
        node_names = {s.node for s in result}
        assert node_names == {"frontend_subgraph", "backend_subgraph"}

    def test_dispatches_single_subgraph(self):
        state = {
            "current_actor": "dispatching",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "pending"},
                {"id": "be_1", "description": "Build API", "domain": "backend", "status": "completed"},
            ],
            "code_base": {},
            "system_design": "React + FastAPI",
        }
        result = route_after_leader(state)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].node == "frontend_subgraph"

    def test_no_pending_tasks_routes_to_end(self):
        state = {
            "current_actor": "dispatching",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "completed"},
            ],
            "code_base": {},
            "system_design": "React + FastAPI",
        }
        result = route_after_leader(state)
        assert result == END

    def test_send_state_contains_filtered_tasks(self):
        """Each Send should contain only tasks for its domain."""
        state = {
            "current_actor": "dispatching",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "pending"},
                {"id": "be_1", "description": "Build API", "domain": "backend", "status": "pending"},
            ],
            "code_base": {"existing.py": "print('hello')"},
            "system_design": "React + FastAPI",
        }
        result = route_after_leader(state)
        for send in result:
            domain = send.arg["domain"]
            for t in send.arg["task_list"]:
                assert t["domain"] == domain, (
                    f"Task {t['id']} (domain={t['domain']}) leaked into {domain} subgraph"
                )


# ── Routing: Domain Subgraph ──────────────────

class TestRouteAfterTaskSelector:
    def test_task_found_routes_to_expert(self):
        state = {"current_task_id": "fe_1"}
        assert _route_after_task_selector(state) == "expert"

    def test_no_task_routes_to_end(self):
        state = {"current_task_id": ""}
        assert _route_after_task_selector(state) == END


class TestRouteAfterTaskReview:
    def test_in_progress_routes_to_expert_retry(self):
        """When task is still in_progress (retry), route back to expert."""
        state = {
            "current_task_id": "fe_1",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "in_progress"},
            ],
        }
        assert _route_after_task_review(state) == "expert"

    def test_completed_routes_to_task_selector(self):
        """When task is completed, route to task_selector for next task."""
        state = {
            "current_task_id": "fe_1",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "completed"},
            ],
        }
        assert _route_after_task_review(state) == "task_selector"

    def test_failed_routes_to_task_selector(self):
        """When task is failed (circuit breaker), route to task_selector."""
        state = {
            "current_task_id": "fe_1",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "failed"},
            ],
        }
        assert _route_after_task_review(state) == "task_selector"

    def test_missing_task_routes_to_task_selector(self):
        """Fallback: if task not found, route to task_selector."""
        state = {
            "current_task_id": "unknown_task",
            "task_list": [],
        }
        assert _route_after_task_review(state) == "task_selector"
