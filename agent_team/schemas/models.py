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
# Leader output
# ──────────────────────────────────────────────

class DispatchedTask(BaseModel):
    """A single task dispatch instruction from the Leader."""

    task_id: str = Field(..., description="ID of the task to dispatch.")
    domain: str = Field(
        ...,
        description="Target expert domain: 'frontend' or 'backend'.",
        pattern=r"^(frontend|backend)$",
    )


class LeaderDecision(BaseModel):
    """Structured output from the Leader agent."""

    dispatched_tasks: list[DispatchedTask] = Field(
        ...,
        description=(
            "List of tasks to dispatch in parallel. "
            "May contain one task per domain (frontend + backend simultaneously)."
        ),
    )
    is_complete: bool = Field(
        default=False,
        description="True if all tasks are completed or failed and no more work remains.",
    )


# ──────────────────────────────────────────────
# Expert output
# ──────────────────────────────────────────────

class ExpertSubmission(BaseModel):
    """Structured output from a Frontend/Backend Expert agent."""

    task_id: str = Field(..., description="ID of the completed sub-task.")
    domain: str = Field(
        ...,
        description="Domain of the expert: 'frontend' or 'backend'.",
        pattern=r"^(frontend|backend)$",
    )
    modified_files: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of file paths to their content that were created or modified.",
    )
    tool_execution_summary: str = Field(
        default="",
        description="Summary of tool calls executed (file reads, writes, bash commands).",
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
