"""
Shared test fixtures — mock LLM responses for testing without an API key.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent_team.schemas.models import (
    TaskItem,
    PlannerOutput,
    LeaderDecision,
    DispatchedTask,
    ExpertSubmission,
    ReviewerEvaluation,
)


@pytest.fixture
def sample_planner_output() -> PlannerOutput:
    """A valid PlannerOutput for testing."""
    return PlannerOutput(
        system_architecture=(
            "Stack: Frontend=React, Backend=FastAPI. "
            "Components: api_server, web_client."
        ),
        frontend_tasks=[
            TaskItem(id="fe_1", description="Create React project scaffold", domain="frontend"),
            TaskItem(id="fe_2", description="Build main App component", domain="frontend"),
        ],
        backend_tasks=[
            TaskItem(id="be_1", description="Create FastAPI server with health endpoint", domain="backend"),
            TaskItem(id="be_2", description="Add CORS middleware", domain="backend"),
        ],
    )


@pytest.fixture
def sample_leader_decision() -> LeaderDecision:
    """A valid LeaderDecision dispatching to both domains."""
    return LeaderDecision(
        dispatched_tasks=[
            DispatchedTask(task_id="fe_1", domain="frontend"),
            DispatchedTask(task_id="be_1", domain="backend"),
        ],
        is_complete=False,
    )


@pytest.fixture
def sample_expert_submission() -> ExpertSubmission:
    """A valid ExpertSubmission."""
    return ExpertSubmission(
        task_id="fe_1",
        domain="frontend",
        modified_files={"src/App.tsx": "export default function App() { return <h1>Hello</h1>; }"},
        tool_execution_summary="write_file(src/App.tsx)",
    )


@pytest.fixture
def sample_reviewer_pass() -> ReviewerEvaluation:
    """A passing ReviewerEvaluation."""
    return ReviewerEvaluation(
        is_passed=True,
        feedback="",
        task_id="fe_1",
        domain="frontend",
    )


@pytest.fixture
def sample_reviewer_fail() -> ReviewerEvaluation:
    """A failing ReviewerEvaluation."""
    return ReviewerEvaluation(
        is_passed=False,
        feedback="Missing error handling in the main component.",
        task_id="fe_1",
        domain="frontend",
    )


@pytest.fixture
def sample_task_list() -> list[dict]:
    """A flat task list with mixed statuses."""
    return [
        {"id": "fe_1", "description": "Create React scaffold", "domain": "frontend", "status": "pending"},
        {"id": "fe_2", "description": "Build App component", "domain": "frontend", "status": "pending"},
        {"id": "be_1", "description": "Create FastAPI server", "domain": "backend", "status": "pending"},
        {"id": "be_2", "description": "Add CORS middleware", "domain": "backend", "status": "pending"},
    ]


@pytest.fixture
def sample_initial_state(sample_task_list) -> dict:
    """A valid initial GraphState dict."""
    return {
        "original_requirement": "Build a simple web app",
        "system_design": "Stack: Frontend=React, Backend=FastAPI",
        "task_list": sample_task_list,
        "current_active_tasks": {},
        "code_base": {},
        "retry_counters": {},
        "current_actor": "planner",
        "review_feedback": "",
        "phase": "planning",
        "expert_submissions": [],
    }
