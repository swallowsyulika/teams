"""
Tests for Pydantic schema models — validates data contracts.
"""

import pytest
from pydantic import ValidationError

from agent_team.schemas.models import (
    TaskItem,
    PlannerOutput,
    LeaderDecision,
    DispatchedTask,
    ExpertSubmission,
    ReviewerEvaluation,
)


# ── TaskItem ───────────────────────────────────

class TestTaskItem:
    def test_valid_frontend_task(self):
        t = TaskItem(id="fe_1", description="Build login page", domain="frontend")
        assert t.status == "pending"
        assert t.domain == "frontend"

    def test_valid_backend_task(self):
        t = TaskItem(id="be_3", description="Setup DB", domain="backend", status="completed")
        assert t.status == "completed"

    def test_invalid_domain_rejected(self):
        with pytest.raises(ValidationError):
            TaskItem(id="x_1", description="Bad", domain="devops")

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            TaskItem(id="fe_1", description="Bad", domain="frontend", status="unknown")


# ── PlannerOutput ──────────────────────────────

class TestPlannerOutput:
    def test_valid_output(self, sample_planner_output):
        assert len(sample_planner_output.frontend_tasks) == 2
        assert len(sample_planner_output.backend_tasks) == 2
        assert "stack" in sample_planner_output.system_architecture

    def test_empty_tasks_allowed(self):
        po = PlannerOutput(
            system_architecture={"desc": "API only"},
            frontend_tasks=[],
            backend_tasks=[TaskItem(id="be_1", description="Create server", domain="backend")],
        )
        assert len(po.frontend_tasks) == 0

    def test_missing_architecture_rejected(self):
        with pytest.raises(ValidationError):
            PlannerOutput(
                frontend_tasks=[],
                backend_tasks=[],
            )


# ── LeaderDecision ─────────────────────────────

class TestLeaderDecision:
    def test_valid_parallel_dispatch(self, sample_leader_decision):
        assert len(sample_leader_decision.dispatched_tasks) == 2
        assert not sample_leader_decision.is_complete

    def test_complete_decision(self):
        d = LeaderDecision(dispatched_tasks=[], is_complete=True)
        assert d.is_complete
        assert len(d.dispatched_tasks) == 0

    def test_invalid_domain_in_dispatch(self):
        with pytest.raises(ValidationError):
            DispatchedTask(task_id="x_1", domain="database")


# ── ExpertSubmission ───────────────────────────

class TestExpertSubmission:
    def test_valid_submission(self, sample_expert_submission):
        assert sample_expert_submission.task_id == "fe_1"
        assert "src/App.tsx" in sample_expert_submission.modified_files

    def test_empty_submission(self):
        s = ExpertSubmission(task_id="be_1", domain="backend")
        assert s.modified_files == {}
        assert s.tool_execution_summary == ""


# ── ReviewerEvaluation ─────────────────────────

class TestReviewerEvaluation:
    def test_pass_evaluation(self, sample_reviewer_pass):
        assert sample_reviewer_pass.is_passed is True

    def test_fail_evaluation(self, sample_reviewer_fail):
        assert sample_reviewer_fail.is_passed is False
        assert "error handling" in sample_reviewer_fail.feedback.lower()

    def test_missing_is_passed_rejected(self):
        with pytest.raises(ValidationError):
            ReviewerEvaluation(feedback="something")
