"""
Tests for graph construction — verifies the StateGraph compiles and
nodes/edges are wired correctly, without requiring an LLM API key.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_team.graph.builder import (
    build_graph,
    route_after_plan_review,
    route_after_leader,
    route_after_task_review,
)
from langgraph.graph import END
from langgraph.types import Send


# ── Graph compilation ──────────────────────────

class TestGraphCompilation:
    def test_graph_compiles_successfully(self):
        """The graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """All 6 node names should be present in the compiled graph."""
        graph = build_graph()
        graph_def = graph.get_graph()
        node_ids = {n.id for n in graph_def.nodes.values()}
        expected = {"planner", "plan_reviewer", "leader",
                    "frontend_expert", "backend_expert", "task_reviewer"}
        assert expected.issubset(node_ids), f"Missing nodes: {expected - node_ids}"


# ── Routing functions ──────────────────────────

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


class TestRouteAfterLeader:
    def test_done_routes_to_end(self):
        state = {"current_actor": "done", "current_active_tasks": {}}
        result = route_after_leader(state)
        assert result == END

    def test_dispatches_both_experts_in_parallel(self):
        state = {
            "current_actor": "dispatching",
            "current_active_tasks": {"frontend": "fe_1", "backend": "be_1"},
        }
        result = route_after_leader(state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)
        node_names = {s.node for s in result}
        assert node_names == {"frontend_expert", "backend_expert"}

    def test_dispatches_single_expert(self):
        state = {
            "current_actor": "dispatching",
            "current_active_tasks": {"frontend": "fe_1"},
        }
        result = route_after_leader(state)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].node == "frontend_expert"

    def test_no_active_tasks_routes_to_end(self):
        state = {
            "current_actor": "dispatching",
            "current_active_tasks": {},
        }
        result = route_after_leader(state)
        assert result == END


class TestRouteAfterTaskReview:
    def test_pass_routes_to_leader_string(self):
        """Single actor 'leader' returns plain string."""
        state = {"current_actor": "leader"}
        assert route_after_task_review(state) == "leader"

    def test_fail_routes_to_frontend_expert_string(self):
        state = {"current_actor": "frontend_expert"}
        assert route_after_task_review(state) == "frontend_expert"

    def test_fail_routes_to_backend_expert_string(self):
        state = {"current_actor": "backend_expert"}
        assert route_after_task_review(state) == "backend_expert"

    def test_unknown_actor_defaults_to_leader(self):
        state = {"current_actor": "unknown"}
        assert route_after_task_review(state) == "leader"

    def test_list_with_experts_uses_send(self):
        """When current_actor is a list with experts, should return Send objects."""
        state = {"current_actor": ["frontend_expert", "backend_expert"]}
        result = route_after_task_review(state)
        assert isinstance(result, list)
        assert all(isinstance(s, Send) for s in result)
        node_names = {s.node for s in result}
        assert node_names == {"frontend_expert", "backend_expert"}

    def test_list_with_only_leader(self):
        """When current_actor is ['leader'], returns plain string 'leader'."""
        state = {"current_actor": ["leader"]}
        assert route_after_task_review(state) == "leader"

    def test_list_with_expert_and_leader_excludes_leader(self):
        """When mixed, only experts are dispatched via Send — leader is excluded
        to prevent race conditions (defense-in-depth alongside reviewer logic)."""
        state = {"current_actor": ["frontend_expert", "leader"]}
        result = route_after_task_review(state)
        assert isinstance(result, list)
        node_names = {s.node for s in result}
        assert node_names == {"frontend_expert"}
        assert "leader" not in node_names
