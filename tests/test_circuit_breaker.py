"""
Tests for the circuit breaker mechanism in the Reviewer.

Verifies that a task reaching MAX_RETRIES is marked 'failed' and
routed back to the leader instead of looping indefinitely.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_team.schemas.models import ReviewerEvaluation
from agent_team.agents.reviewer import reviewer_node
from agent_team.graph.config import MAX_RETRIES


def _make_execution_state(
    task_id: str = "fe_1",
    domain: str = "frontend",
    retry_count: int = 0,
) -> dict:
    """Create a state dict simulating an execution-phase review."""
    return {
        "original_requirement": "Build a web app",
        "system_design": {"stack": "React + FastAPI"},
        "task_list": [
            {"id": task_id, "description": "Build component", "domain": domain, "status": "in_progress"},
            {"id": "be_1", "description": "Build API", "domain": "backend", "status": "pending"},
        ],
        "current_active_tasks": {domain: task_id},
        "code_base": {},
        "retry_counters": {task_id: retry_count},
        "current_actor": "task_reviewer",
        "review_feedback": "",
        "phase": "execution",
        "expert_submissions": [
            {
                "task_id": task_id,
                "domain": domain,
                "modified_files": {"src/Component.tsx": "export default () => <div/>;"},
                "tool_execution_summary": "write_file(src/Component.tsx)",
            }
        ],
    }


class TestCircuitBreaker:
    @patch("agent_team.agents.reviewer.ChatOpenAI")
    def test_normal_rejection_routes_to_expert(self, mock_chat_cls):
        """When retry count < MAX_RETRIES, rejection routes back to expert."""
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

        state = _make_execution_state(retry_count=0)
        result = reviewer_node(state)

        assert result["current_actor"] == "frontend_expert"
        assert result["retry_counters"]["fe_1"] == 1
        assert "Missing error boundary" in result["review_feedback"]

        # Task should still be in_progress
        fe_task = next(t for t in result["task_list"] if t["id"] == "fe_1")
        assert fe_task["status"] == "in_progress"

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

        # Set retry count to MAX_RETRIES - 1 (will become MAX_RETRIES after increment)
        state = _make_execution_state(retry_count=MAX_RETRIES - 1)
        result = reviewer_node(state)

        # Should route to leader, not expert
        assert result["current_actor"] == "leader"
        assert result["retry_counters"]["fe_1"] == MAX_RETRIES

        # Task should be marked as failed
        fe_task = next(t for t in result["task_list"] if t["id"] == "fe_1")
        assert fe_task["status"] == "failed"

        # Feedback should mention circuit breaker
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

        state = _make_execution_state(retry_count=0)
        result = reviewer_node(state)

        assert result["current_actor"] == "leader"
        fe_task = next(t for t in result["task_list"] if t["id"] == "fe_1")
        assert fe_task["status"] == "completed"

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

        state = {
            "original_requirement": "Build an app",
            "system_design": {"stack": "React"},
            "task_list": [{"id": "fe_1", "description": "Build UI", "domain": "frontend", "status": "pending"}],
            "phase": "planning",
            "review_feedback": "",
            "retry_counters": {},
            "current_active_tasks": {},
            "code_base": {},
            "expert_submissions": [],
        }
        result = reviewer_node(state)

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

        state = {
            "original_requirement": "Build an app",
            "system_design": {"stack": "React"},
            "task_list": [{"id": "fe_1", "description": "Build entire frontend", "domain": "frontend", "status": "pending"}],
            "phase": "planning",
            "review_feedback": "",
            "retry_counters": {},
            "current_active_tasks": {},
            "code_base": {},
            "expert_submissions": [],
        }
        result = reviewer_node(state)

        assert result["current_actor"] == "planner"
        assert "coarse-grained" in result["review_feedback"].lower()
