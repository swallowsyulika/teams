"""
Pydantic structured-output models for inter-agent communication.

Each model enforces the data contract between agents, making LLM outputs
parseable and type-safe.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Task representation
# ──────────────────────────────────────────────

class TaskItem(BaseModel):
    """A single fine-grained sub-task."""

    id: str = Field(..., description="Unique task identifier, e.g. 'fe_1', 'be_3'.")
    description: str = Field(..., description="Clear, actionable description of the task.")
    domain: str = Field(
        ...,
        description="Domain this task belongs to: 'frontend' or 'backend'.",
        pattern=r"^(frontend|backend)$",
    )
    status: str = Field(
        default="pending",
        description="Current status: pending | in_progress | completed | failed.",
        pattern=r"^(pending|in_progress|completed|failed)$",
    )


# ──────────────────────────────────────────────
# Planner output
# ──────────────────────────────────────────────

class PlannerOutput(BaseModel):
    """Structured output from the Planner agent."""

    system_architecture: str = Field(
        ...,
        description=(
            "High-level architecture design including technology stack, "
            "component breakdown, data flow, and API contracts."
        ),
    )
    frontend_tasks: list[TaskItem] = Field(
        ...,
        description="High-level frontend sub-tasks focusing purely on coding (e.g. pages, components).",
    )
    backend_tasks: list[TaskItem] = Field(
        ...,
        description="High-level backend sub-tasks focusing purely on coding (e.g. APIs, features).",
    )



# ──────────────────────────────────────────────
# Reviewer output
# ──────────────────────────────────────────────

class ReviewerEvaluation(BaseModel):
    """Structured output from the Reviewer agent."""

    is_passed: bool = Field(
        ...,
        description="True if the submission meets quality standards.",
    )
    feedback: str = Field(
        default="",
        description="Detailed feedback. Required when is_passed is False.",
    )
    task_id: str = Field(
        default="",
        description="ID of the reviewed task (empty during Phase 1 plan review).",
    )
    domain: str = Field(
        default="",
        description="Domain of the reviewed task (empty during Phase 1 plan review).",
    )
