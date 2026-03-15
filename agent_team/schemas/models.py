"""
Pydantic structured-output models for inter-agent communication.

Each model enforces the data contract between agents, making LLM outputs
parseable and type-safe.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from agent_team.graph.config import ENABLED_EXPERTS


# ──────────────────────────────────────────────
# Task representation
# ──────────────────────────────────────────────

def _get_task_item_model() -> type[BaseModel]:
    domain_pattern = f"^({'|'.join(ENABLED_EXPERTS)})$"
    
    class _TaskItem(BaseModel):
        """A single fine-grained sub-task."""

        id: str = Field(..., description="Unique task identifier, e.g. 'fe_1', 'be_3'.")
        description: str = Field(..., description="Clear, actionable description of the task.")
        domain: str = Field(
            ...,
            description=f"Domain this task belongs to: {', '.join(ENABLED_EXPERTS)}.",
            pattern=domain_pattern,
        )
        status: str = Field(
            default="pending",
            description="Current status: pending | in_progress | completed | failed.",
            pattern=r"^(pending|in_progress|completed|failed)$",
        )
    return _TaskItem


TaskItem = _get_task_item_model()


# ──────────────────────────────────────────────
# Planner output
# ──────────────────────────────────────────────

def _get_planner_output_model() -> type[BaseModel]:
    fields = {
        "system_architecture": (
            str,
            Field(
                ...,
                description=(
                    "High-level architecture design including technology stack, "
                    "component breakdown, data flow, and API contracts."
                ),
            )
        )
    }
    
    for domain in ENABLED_EXPERTS:
        fields[f"{domain}_tasks"] = (
            list[TaskItem],
            Field(
                ...,
                description=f"High-level {domain} sub-tasks focusing purely on coding.",
            )
        )
        
    return create_model("PlannerOutput", **fields)


PlannerOutput = _get_planner_output_model()



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
