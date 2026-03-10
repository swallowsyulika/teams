"""
Reviewer agent node — quality gate for both phases.

Phase 1 (planning): Reviews system design and task granularity.
Phase 2 (execution): Reviews expert code submissions for correctness.
Implements the circuit-breaker mechanism (MAX_RETRIES).
"""

from __future__ import annotations

import json
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

PLAN_REVIEWER_SYSTEM_PROMPT = """\
You are the **Reviewer** for a multi-agent software development team.

You are reviewing the **Planner's system design and task breakdown**.

Check for:
1. Architecture completeness — does it address the user requirement?
2. Task granularity — is each sub-task small enough to be completed in
   a single LLM generation step?
3. Logical consistency — are there missing tasks, circular dependencies,
   or contradictions?
4. Frontend/backend separation — are tasks correctly assigned?

Be strict but fair.  If you find ANY issue, set is_passed=false and
provide specific, actionable feedback.  Never compromise.
"""

TASK_REVIEWER_SYSTEM_PROMPT = """\
You are the **Reviewer** for a multi-agent software development team.

You are reviewing an **Expert's code submission** for a specific sub-task.

Check for:
1. Task completion — does the code fully implement the task description?
2. Logic errors — are there bugs, off-by-one errors, missing edge cases?
3. Code quality — is the code clean, well-structured, and maintainable?
4. Security — is there any malicious or dangerous code?
5. File organization — are files placed correctly?

Be strict but fair.  If you find ANY issue, set is_passed=false and
provide specific, actionable feedback.  Never compromise.
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
    system_design = state.get("system_design", {})
    task_list = state.get("task_list", [])
    requirement = state.get("original_requirement", "")

    messages = [
        SystemMessage(content=PLAN_REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Original Requirement\n{requirement}\n\n"
                f"## System Architecture\n```json\n{json.dumps(system_design, indent=2)}\n```\n\n"
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
        return {
            "current_actor": "planner",
            "review_feedback": evaluation.feedback,
        }


def _review_task(state: GraphState, llm) -> dict[str, Any]:
    """Review an Expert's code submission."""
    submissions = state.get("expert_submissions", [])
    task_list = state.get("task_list", [])
    retry_counters = dict(state.get("retry_counters", {}))
    system_design = state.get("system_design", {})
    code_base = state.get("code_base", {})

    if not submissions:
        # No submission to review — route back to leader
        return {"current_actor": "leader"}

    # Review the most recent submission
    submission = submissions[-1]
    task_id = submission.get("task_id", "")
    domain = submission.get("domain", "")
    modified_files = submission.get("modified_files", {})
    tool_summary = submission.get("tool_execution_summary", "")

    # Find the task description
    task_desc = ""
    for t in task_list:
        if t["id"] == task_id:
            task_desc = t["description"]
            break

    messages = [
        SystemMessage(content=TASK_REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Task\n- **ID**: {task_id}\n- **Domain**: {domain}\n"
                f"- **Description**: {task_desc}\n\n"
                f"## System Architecture\n```json\n{json.dumps(system_design, indent=2)}\n```\n\n"
                f"## Modified Files\n"
                + (
                    "\n".join(
                        f"### `{fp}`\n```\n{content}\n```"
                        for fp, content in modified_files.items()
                    )
                    if modified_files
                    else "(no files modified)\n"
                )
                + f"\n## Tool Execution Log\n{tool_summary}"
            )
        ),
    ]

    evaluation: ReviewerEvaluation = llm.invoke(messages)

    # Update task list
    updated_tasks = []
    for t in task_list:
        t_copy = dict(t)
        if t_copy["id"] == task_id:
            if evaluation.is_passed:
                t_copy["status"] = "completed"
            # On fail, status stays "in_progress" for retry
        updated_tasks.append(t_copy)

    if evaluation.is_passed:
        return {
            "task_list": updated_tasks,
            "current_actor": "leader",
            "review_feedback": "",
        }

    # ── Failure path ──
    retry_count = retry_counters.get(task_id, 0) + 1
    retry_counters[task_id] = retry_count

    # Circuit breaker
    if retry_count >= MAX_RETRIES:
        # Mark task as failed and route to leader
        for t in updated_tasks:
            if t["id"] == task_id:
                t["status"] = "failed"
                break
        return {
            "task_list": updated_tasks,
            "retry_counters": retry_counters,
            "current_actor": "leader",
            "review_feedback": (
                f"CIRCUIT BREAKER: Task {task_id} failed after "
                f"{MAX_RETRIES} retries. Marked as failed.\n"
                f"Last feedback: {evaluation.feedback}"
            ),
        }

    # Normal rejection — return to the originating expert
    actor = f"{domain}_expert" if domain else "leader"
    return {
        "task_list": updated_tasks,
        "retry_counters": retry_counters,
        "current_actor": actor,
        "review_feedback": evaluation.feedback,
    }
