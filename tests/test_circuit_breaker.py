"""
Tests for the circuit breaker mechanism.

Phase 1: Planner circuit breaker (plan_reviewer_node).
Phase 2: Task circuit breaker (task_reviewer_node inside subgraph).
Also tests the task_selector_node for queue exhaustion.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_team.schemas.models import ReviewerEvaluation
from agent_team.agents.reviewer import plan_reviewer_node, task_reviewer_node
from agent_team.agents.experts import task_selector_node
from agent_team.graph.config import MAX_RETRIES


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_domain_state(
    task_id: str = "fe_1",
    domain: str = "frontend",
    retry_count: int = 0,
    task_status: str = "in_progress",
) -> dict:
    """Create a DomainState dict simulating a subgraph execution."""
    return {
        "domain": domain,
        "task_list": [
            {"id": task_id, "description": "Build component", "domain": domain, "status": task_status},
            {"id": "fe_2", "description": "Build page", "domain": domain, "status": "pending"},
        ],
        "code_base": {"src/Component.tsx": "export default () => <div/>;"},
        "system_design": "React + FastAPI",
        "current_task_id": task_id,
        "review_feedback": "",
        "retry_count": retry_count,
    }


def _make_planning_state(retry_count: int = 0) -> dict:
    """Create a state dict simulating a planning-phase review."""
    return {
        "original_requirement": "Build an app",
        "system_design": "Stack: React + FastAPI",
        "task_list": [
            {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "pending"}
        ],
        "phase": "planning",
        "review_feedback": "",
        "retry_counters": {"planning": retry_count},
        "code_base": {},
    }


# ──────────────────────────────────────────────
# Task Selector Tests
# ──────────────────────────────────────────────

class TestTaskSelector:
    def test_picks_first_pending_task(self):
        """Task selector picks the first pending task for the domain."""
        state = {
            "domain": "frontend",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "completed"},
                {"id": "fe_2", "description": "Build page", "domain": "frontend", "status": "pending"},
            ],
        }
        result = task_selector_node(state)
        assert result["current_task_id"] == "fe_2"
        # Should mark the picked task as in_progress
        task = next(t for t in result["task_list"] if t["id"] == "fe_2")
        assert task["status"] == "in_progress"

    def test_returns_empty_when_queue_exhausted(self):
        """Task selector returns empty current_task_id when no pending tasks."""
        state = {
            "domain": "frontend",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "completed"},
                {"id": "fe_2", "description": "Build page", "domain": "frontend", "status": "failed"},
            ],
        }
        result = task_selector_node(state)
        assert result["current_task_id"] == ""

    def test_ignores_other_domain_tasks(self):
        """Task selector only considers tasks for its own domain."""
        state = {
            "domain": "frontend",
            "task_list": [
                {"id": "be_1", "description": "Build API", "domain": "backend", "status": "pending"},
            ],
        }
        result = task_selector_node(state)
        assert result["current_task_id"] == ""

    def test_resets_retry_count_and_feedback(self):
        """Task selector resets retry_count and review_feedback for the new task."""
        state = {
            "domain": "frontend",
            "task_list": [
                {"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "pending"},
            ],
        }
        result = task_selector_node(state)
        assert result["retry_count"] == 0
        assert result["review_feedback"] == ""


# ──────────────────────────────────────────────
# Task Circuit Breaker Tests (Subgraph)
# ──────────────────────────────────────────────

class TestTaskCircuitBreaker:
    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_normal_rejection_keeps_in_progress(self, mock_chat_cls):
        """When retry count < MAX_RETRIES, rejection keeps task in_progress."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=False,
            feedback="Missing error boundary.",
            task_id="fe_1",
            domain="frontend",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_domain_state(retry_count=0)
        result = task_reviewer_node(state)

        assert result["retry_count"] == 1
        assert "fe_1" in result["review_feedback"]
        # Task list should NOT be returned (task stays in_progress)
        assert "task_list" not in result

    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_circuit_breaker_triggers_at_max_retries(self, mock_chat_cls):
        """When retry count reaches MAX_RETRIES, task is marked failed."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=False,
            feedback="Still broken.",
            task_id="fe_1",
            domain="frontend",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_domain_state(retry_count=MAX_RETRIES - 1)
        result = task_reviewer_node(state)

        assert result["retry_count"] == MAX_RETRIES
        # Task should be marked as failed
        fe_task = next(t for t in result["task_list"] if t["id"] == "fe_1")
        assert fe_task["status"] == "failed"
        assert "CIRCUIT BREAKER" in result["review_feedback"]

    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_pass_marks_task_completed(self, mock_chat_cls):
        """When reviewer passes, task is marked completed."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=True,
            feedback="",
            task_id="fe_1",
            domain="frontend",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_domain_state(retry_count=0)
        result = task_reviewer_node(state)

        fe_task = next(t for t in result["task_list"] if t["id"] == "fe_1")
        assert fe_task["status"] == "completed"
        assert result["review_feedback"] == ""

    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_circuit_break_allows_next_task(self, mock_chat_cls):
        """After circuit break fails a task, the queue still has pending tasks."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=False,
            feedback="Unfixable.",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_domain_state(retry_count=MAX_RETRIES - 1)
        result = task_reviewer_node(state)

        # fe_1 is failed, but fe_2 should still be pending
        fe_1 = next(t for t in result["task_list"] if t["id"] == "fe_1")
        fe_2 = next(t for t in result["task_list"] if t["id"] == "fe_2")
        assert fe_1["status"] == "failed"
        assert fe_2["status"] == "pending"


# ──────────────────────────────────────────────
# Plan Circuit Breaker Tests (Main Graph)
# ──────────────────────────────────────────────

class TestPlanCircuitBreaker:
    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_plan_review_pass(self, mock_chat_cls):
        """Phase 1 plan review pass routes to leader and sets phase to execution."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=True,
            feedback="",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_planning_state()
        result = plan_reviewer_node(state)

        assert result["current_actor"] == "leader"
        assert result["phase"] == "execution"

    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_plan_review_fail_routes_to_planner(self, mock_chat_cls):
        """Phase 1 plan review fail routes back to planner with feedback."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=False,
            feedback="Tasks are too coarse-grained.",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_planning_state()
        result = plan_reviewer_node(state)

        assert result["current_actor"] == "planner"
        assert "coarse-grained" in result["review_feedback"].lower()
        assert result["retry_counters"]["planning"] == 1

    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_plan_circuit_breaker(self, mock_chat_cls):
        """Phase 1 plan review circuit breaker after MAX_RETRIES."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ReviewerEvaluation(
            is_passed=False,
            feedback="Still bad.",
        )
        mock_chat_instance = MagicMock()
        mock_chat_instance.with_structured_output.return_value = mock_llm
        mock_chat_cls.return_value = mock_chat_instance

        state = _make_planning_state(retry_count=MAX_RETRIES - 1)
        result = plan_reviewer_node(state)

        assert result["current_actor"] == "done"
        assert "CIRCUIT BREAKER" in result["review_feedback"]
        assert result["retry_counters"]["planning"] == MAX_RETRIES
