"""
Reviewer agent node — quality gate for both phases.

Phase 1 (planning): Reviews system design and task granularity.
Phase 2 (execution): Reviews expert code submissions for correctness.
Implements the circuit-breaker mechanism (MAX_RETRIES).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_team.graph.config import (
    MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    TEMPERATURE,
    MAX_RETRIES,
)
from agent_team.schemas.models import ReviewerEvaluation
from agent_team.schemas.state import GraphState

# Maximum characters of file content to include in a review prompt per file.
# Keeps the reviewer LLM context bounded even for large generated files.
_MAX_FILE_CHARS = 8_000

PLAN_REVIEWER_SYSTEM_PROMPT = """\
You are the **Reviewer** for a multi-agent software development team.

You are reviewing the **Planner's system design and task breakdown**.

Check for:
1. Requirement fulfillment — are the generated tasks collectively sufficient to completely fulfill the user requirements (URD)?
2. Task focus and granularity — do the tasks focus ONLY on source code implementation (e.g., pages, components, APIs, features)? Reject tasks related to environment setup, Dockerfiles, CI/CD, deployment, or documentation. Tasks should be high-level directional goals, not overly fine-grained.
3. Logical consistency — is the direction of each task correct? Are there missing coding tasks, circular dependencies, or contradictions?
4. Frontend/backend separation — are tasks correctly assigned?

Be strict but fair. If you find ANY issue, set is_passed=false and
provide specific, actionable feedback. Never compromise.
"""

TASK_REVIEWER_SYSTEM_PROMPT = """\
You are the **Reviewer** for a multi-agent software development team.

You are reviewing an **Expert's code submission** for a specific sub-task.

Check for:
1. Task completion — does the source code fully implement the task description?
2. Code correctness — verify the logic, parameters, and ensure there are no bugs.
3. Security — check for malicious code or dangerous injections.
4. Pure coding focus — the submission must only contain application source code. No need to check for environment setups, infrastructure files, or deployments.

Be strict but fair. If you find ANY issue, set is_passed=false and
provide specific, actionable feedback. Never compromise.
"""


def reviewer_node(state: GraphState) -> dict[str, Any]:
    """Reviewer node — validates planner output or expert submissions.

    Routing behaviour is determined by ``phase`` and the evaluation result:
    - Phase "planning": pass → leader, fail → planner
    - Phase "execution": pass → leader (mark task completed),
      fail → back to originating expert, circuit-break → leader (mark failed)

    Args:
        state: Current graph state.

    Returns:
        Partial state update with review results and routing info.
    """
    phase = state.get("phase", "planning")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    ).with_structured_output(ReviewerEvaluation)

    if phase == "planning":
        return _review_plan(state, llm)
    else:
        return _review_task(state, llm)


def _review_plan(state: GraphState, llm) -> dict[str, Any]:
    """Review the Planner's system design and task breakdown."""
    system_design = state.get("system_design", "")
    task_list = state.get("task_list", [])
    requirement = state.get("original_requirement", "")

    messages = [
        SystemMessage(content=PLAN_REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Original Requirement\n{requirement}\n\n"
                f"## System Architecture\n{system_design}\n\n"
                f"## Task List ({len(task_list)} tasks)\n"
                + "\n".join(
                    f"- {t['id']} [{t['domain']}]: {t['description']}"
                    for t in task_list
                )
            )
        ),
    ]

    evaluation: ReviewerEvaluation = llm.invoke(messages)

    if evaluation.is_passed:
        return {
            "phase": "execution",
            "current_actor": "leader",
            "review_feedback": "",
        }
    else:
        retry_counters = dict(state.get("retry_counters", {}))
        retry_count = retry_counters.get("planning", 0) + 1
        retry_counters["planning"] = retry_count

        if retry_count >= MAX_RETRIES:
            return {
                "retry_counters": retry_counters,
                "current_actor": "done",
                "review_feedback": f"CIRCUIT BREAKER: Planner failed after {MAX_RETRIES} retries.",
            }

        return {
            "retry_counters": retry_counters,
            "current_actor": "planner",
            "review_feedback": evaluation.feedback,
        }


def _review_task(state: GraphState, llm) -> dict[str, Any]:
    """Review an Expert's code submission."""
    submissions = state.get("expert_submissions", [])
    task_list = state.get("task_list", [])
    retry_counters = dict(state.get("retry_counters", {}))
    system_design = state.get("system_design", "")
    code_base = state.get("code_base", {})

    if not submissions:
        # No submission to review — route back to leader
        return {"current_actor": "leader"}

    active_tasks = dict(state.get("current_active_tasks", {}))
    active_ids = set(active_tasks.values())

    # Find the most recent unreviewed submission for each active task
    latest_subs = {}
    for sub in reversed(submissions):
        tid = sub.get("task_id", "")
        if tid in active_ids and tid not in latest_subs:
            latest_subs[tid] = sub

    if not latest_subs:
        return {"current_actor": "leader"}

    updated_tasks = [dict(t) for t in task_list]
    next_actors = []
    feedbacks = []

    for sub in latest_subs.values():
        task_id = sub.get("task_id", "")
        domain = sub.get("domain", "")
        modified_files = sub.get("modified_files", {})
        tool_summary = sub.get("tool_execution_summary", "")

        # Check if task is already completed or failed (avoid re-reviewing)
        task_status = "pending"
        task_desc = ""
        for t in updated_tasks:
            if t["id"] == task_id:
                task_desc = t["description"]
                task_status = t["status"]
                break
                
        if task_status in ("completed", "failed"):
            continue

        # Compact system_design to save tokens
        design_text = system_design
        if len(design_text) > 4000:
            design_text = design_text[:4000] + '... (truncated)'

        # Truncate large file contents to keep LLM context bounded
        file_sections = []
        for fp, content in modified_files.items():
            if len(content) > _MAX_FILE_CHARS:
                content = content[:_MAX_FILE_CHARS] + f"\n... (truncated, {len(content)} chars total)"
            file_sections.append(f"### `{fp}`\n```\n{content}\n```")

        messages = [
            SystemMessage(content=TASK_REVIEWER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"## Task\n- **ID**: {task_id}\n- **Domain**: {domain}\n"
                    f"- **Description**: {task_desc}\n\n"
                    f"## System Architecture\n{design_text}\n\n"
                    f"## Modified Files\n"
                    + (
                        "\n".join(file_sections)
                        if file_sections
                        else "(no files modified)\n"
                    )
                    + f"\n## Tool Execution Log\n{tool_summary[:3000]}"
                )
            ),
        ]

        evaluation: ReviewerEvaluation = llm.invoke(messages)

        if evaluation.is_passed:
            for t in updated_tasks:
                if t["id"] == task_id:
                    t["status"] = "completed"
                    break
            next_actors.append("leader")
        else:
            retry_count = retry_counters.get(task_id, 0) + 1
            retry_counters[task_id] = retry_count

            if retry_count >= MAX_RETRIES:
                for t in updated_tasks:
                    if t["id"] == task_id:
                        t["status"] = "failed"
                        break
                next_actors.append("leader")
                feedbacks.append(f"CIRCUIT BREAKER on {task_id}: {evaluation.feedback}")
            else:
                actor = f"{domain}_expert" if domain else "leader"
                next_actors.append(actor)
                feedbacks.append(f"[{task_id}]: {evaluation.feedback}")

    if not next_actors:
        return {"current_actor": "leader"}

    unique_actors = list(set(next_actors))

    # If any expert needs a retry, route ONLY to the expert(s).
    # The leader must NOT run in the same super-step as a retrying
    # expert, because it would dispatch new tasks on stale state.
    expert_actors = [a for a in unique_actors if a != "leader"]
    if expert_actors:
        # Experts need retries — defer leader until next review cycle
        final_actors = expert_actors
    else:
        # All tasks in this round passed or circuit-broke — wake leader
        final_actors = ["leader"]

    # Collapse single-element list to plain string for simpler routing
    current_actor = final_actors[0] if len(final_actors) == 1 else final_actors

    return {
        "task_list": updated_tasks,
        "retry_counters": retry_counters,
        "current_actor": current_actor,
        "review_feedback": "\n\n".join(feedbacks),
    }
