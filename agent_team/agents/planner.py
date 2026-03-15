"""
Planner agent node — Phase 1.

Receives the original requirement and produces a system architecture
with fine-grained frontend/backend task lists.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_team.graph.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE, ENABLED_EXPERTS
from agent_team.schemas.models import PlannerOutput
from agent_team.schemas.state import GraphState

PLANNER_SYSTEM_PROMPT = f"""\
You are the **Planner** of a multi-agent software development team.

Your job:
1. Analyse the user's software requirement (URD).
2. Design a clear, modular system architecture (technology stack, component
   breakdown, data flow, API contracts).
3. Break the implementation into a **high-level** list of sub-tasks focusing ONLY on coding.
   - A task should represent a broad goal such as: "a whole page", "a complex component", "an API endpoint", or "a core feature".
   - Do NOT create overly fine-grained tasks.
   - Do NOT create tasks for environment setup, Dockerfiles, CI/CD pipelines, configurations, or documentation.
4. Separate tasks into the following domains: {', '.join(ENABLED_EXPERTS)}.
5. Each task must have a unique ID (e.g. "fe_1", "be_3"), a clear description,
   and be marked with status="pending".

Output your result as structured JSON matching the PlannerOutput schema.
If the reviewer previously rejected your plan, their feedback is included
below — revise accordingly.
"""


def planner_node(state: GraphState) -> dict[str, Any]:
    """Planner node — generates system design and task breakdown.

    Args:
        state: Current graph state.

    Returns:
        Partial state update with system_design, task_list, phase, current_actor.
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    ).with_structured_output(PlannerOutput)

    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)]

    # Include original requirement
    requirement = state.get("original_requirement", "")
    messages.append(HumanMessage(content=f"## User Requirement\n\n{requirement}"))

    # If there's reviewer feedback from a previous rejection, include it
    feedback = state.get("review_feedback", "")
    if feedback:
        messages.append(
            HumanMessage(
                content=f"## Reviewer Feedback (please revise your plan)\n\n{feedback}"
            )
        )

    result: PlannerOutput = llm.invoke(messages)

    # Build the flat task list dynamically based on enabled domains
    all_tasks = []
    for domain in ENABLED_EXPERTS:
        tasks = getattr(result, f"{domain}_tasks", [])
        # Each task might be a dict or a Pydantic object, handle both safely
        all_tasks.extend(
            [t.model_dump() if hasattr(t, "model_dump") else t for t in tasks]
        )

    return {
        "system_design": result.system_architecture,
        "task_list": all_tasks,
        "phase": "planning",
        "current_actor": "plan_reviewer",
        "review_feedback": "",
        "code_base": state.get("code_base", {}),
    }
