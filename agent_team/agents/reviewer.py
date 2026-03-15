"""
Reviewer agent node — quality gate for both phases.

Phase 1 (planning): Reviews system design and task granularity.
    Runs in the **main graph** as ``plan_reviewer``.

Phase 2 (execution): Reviews expert code submissions for correctness.
    Runs inside **domain subgraphs** as ``task_reviewer``.
    Implements the circuit-breaker mechanism (MAX_RETRIES).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import os

from agent_team.graph.config import (
    MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    TEMPERATURE,
    MAX_RETRIES,
    WORKSPACE_PATH,
)
from agent_team.schemas.models import ReviewerEvaluation
from agent_team.schemas.state import GraphState, DomainState

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


# ──────────────────────────────────────────────
# Phase 1: Plan reviewer (main graph node)
# ──────────────────────────────────────────────

def plan_reviewer_node(state: GraphState) -> dict[str, Any]:
    """Review the Planner's system design and task breakdown.

    This node runs in the **main graph** (Phase 1).

    Routing:
        pass → leader (phase switches to execution)
        fail → planner (with feedback)
        circuit-break → done
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    ).with_structured_output(ReviewerEvaluation)

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


# ──────────────────────────────────────────────
# Phase 2: Task reviewer (domain subgraph node)
# ──────────────────────────────────────────────

def task_reviewer_node(state: DomainState) -> dict[str, Any]:
    """Review an Expert's code submission within a domain subgraph.

    Reads ``current_task_id`` and evaluates the code in ``code_base``.

    Routing (via conditional edge in the subgraph):
        pass → task_selector (to grab next task)
        fail (retries left) → expert (retry with feedback)
        fail (circuit-break) → task_selector (skip this task)
    """
    domain = state.get("domain", "")
    task_id = state.get("current_task_id", "")
    task_list = state.get("task_list", [])
    code_base = state.get("code_base", {})
    system_design = state.get("system_design", "")
    retry_count = state.get("retry_count", 0)

    if not task_id:
        return {}

    # Find the task description
    task_desc = ""
    for t in task_list:
        if t["id"] == task_id:
            task_desc = t["description"]
            break

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    ).with_structured_output(ReviewerEvaluation)

    # Compact system_design to save tokens
    design_text = system_design
    if len(design_text) > 4000:
        design_text = design_text[:4000] + '... (truncated)'

    # Gather relevant files from the physical workspace for this domain.
    # Exclude non-code node_modules and builds to just review the source code.
    file_sections = []
    
    _IGNORE_EXTS = {".lock", ".png", ".jpg", ".jpeg", ".ico", ".svg", ".pyc", ".db", ".sqlite", ".pdf"}
    _IGNORE_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pipfile.lock"}
    _IGNORE_DIRS = {"node_modules", "venv", ".venv", "__pycache__", ".git", "dist", "build", "coverage", ".next", ".nuxt"}

    domain_dir = WORKSPACE_PATH / domain
    if domain_dir.exists():
        for root, dirs, files in os.walk(domain_dir):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS and not d.startswith(".")]
            for file in files:
                if file in _IGNORE_FILES:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in _IGNORE_EXTS:
                    continue
                
                fp = os.path.join(root, file)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    rel_fp = os.path.relpath(fp, start=WORKSPACE_PATH)
                    rel_fp_str = str(rel_fp).replace("\\", "/")
                    
                    if len(content) > _MAX_FILE_CHARS:
                        content = (
                            content[:_MAX_FILE_CHARS]
                            + f"\n... (truncated, {len(content)} chars total)"
                        )
                    file_sections.append(f"### `{rel_fp_str}`\n```\n{content}\n```")
                except Exception:
                    # Ignore unreadable/binary files
                    pass

    messages = [
        SystemMessage(content=TASK_REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Task\n- **ID**: {task_id}\n- **Domain**: {domain}\n"
                f"- **Description**: {task_desc}\n\n"
                f"## System Architecture\n{design_text}\n\n"
                f"## Code Base Files\n"
                + (
                    "\n".join(file_sections)
                    if file_sections
                    else "(no files in code base)\n"
                )
            )
        ),
    ]

    evaluation: ReviewerEvaluation = llm.invoke(messages)

    updated_tasks = [dict(t) for t in task_list]

    if evaluation.is_passed:
        # Mark task as completed
        for t in updated_tasks:
            if t["id"] == task_id:
                t["status"] = "completed"
                break
        return {
            "task_list": updated_tasks,
            "review_feedback": "",
        }
    else:
        new_retry_count = retry_count + 1

        if new_retry_count >= MAX_RETRIES:
            # Circuit breaker: mark task as failed, move on
            for t in updated_tasks:
                if t["id"] == task_id:
                    t["status"] = "failed"
                    break
            return {
                "task_list": updated_tasks,
                "retry_count": new_retry_count,
                "review_feedback": f"CIRCUIT BREAKER on {task_id}: {evaluation.feedback}",
            }
        else:
            # Retry: keep task in_progress, feed back to expert
            return {
                "retry_count": new_retry_count,
                "review_feedback": f"[{task_id}]: {evaluation.feedback}",
            }
