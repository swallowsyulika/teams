"""
Expert agent nodes — Frontend & Backend workers.

Each expert receives a single sub-task (via the subgraph's
``current_task_id``), uses tools (read_file, write_file, bash) via a
ReAct agent loop, then submits the result for review.

Also contains the ``task_selector_node`` used inside domain subgraphs to
pick the next pending task from the queue.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent_team.graph.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from agent_team.schemas.models import ExpertSubmission
from agent_team.schemas.state import DomainState
from agent_team.tools.file_tools import read_file, write_file
from agent_team.tools.bash_tool import bash

EXPERT_TOOLS = [read_file, write_file, bash]

# Safety cap: maximum number of LLM ↔ tool round-trips per invocation
MAX_TOOL_ROUNDS = 10

_EXPERT_SYSTEM_PROMPT_TEMPLATE = """\
You are the **{domain} Expert** of a multi-agent software development team.

You receive ONE sub-task at a time. Your job:
1. Read the task description carefully.
2. Understand the system architecture and existing code base.
3. Use the available tools (read_file, write_file, bash) to implement
   the task completely.
   - All newly created files and source code MUST be placed within the `./{domain}` directory (e.g., `./frontend` or `./backend`).
   - You MUST ONLY generate application source code. Do NOT generate or modify environment files, Dockerfiles, CI/CD scripts, or infrastructure configurations.
4. When you finish, simply output a concluding summary message explaining what you did.

Be thorough but concise. Write clean, production-quality code.
If the reviewer previously rejected your work, their feedback is included —
fix the issues they raised.
"""

# Pre-build the tool-name → tool-function mapping (avoids rebuilding each call)
_TOOL_MAP = {t.name: t for t in EXPERT_TOOLS}


# ──────────────────────────────────────────────
# Task Selector (subgraph node)
# ──────────────────────────────────────────────

def task_selector_node(state: DomainState) -> dict[str, Any]:
    """Pick the next pending task from this domain's queue.

    Scans ``task_list`` for the first task whose ``status`` is
    ``"pending"`` and whose ``domain`` matches the subgraph's domain.

    Returns:
        Partial DomainState with ``current_task_id`` set (empty string
        if the queue is exhausted) and reset ``retry_count`` /
        ``review_feedback``.
    """
    domain = state.get("domain", "")
    task_list = state.get("task_list", [])

    for t in task_list:
        if t["domain"] == domain and t["status"] == "pending":
            # Mark as in_progress
            updated_tasks = []
            for task in task_list:
                t_copy = dict(task)
                if t_copy["id"] == t["id"]:
                    t_copy["status"] = "in_progress"
                updated_tasks.append(t_copy)

            return {
                "current_task_id": t["id"],
                "task_list": updated_tasks,
                "retry_count": 0,
                "review_feedback": "",
            }

    # No pending tasks left for this domain
    return {
        "current_task_id": "",
    }


# ──────────────────────────────────────────────
# Expert node (subgraph node)
# ──────────────────────────────────────────────

def expert_node(state: DomainState) -> dict[str, Any]:
    """Expert node — implements a single sub-task using tools.

    Runs a full ReAct loop: the LLM is called repeatedly, executing any
    requested tool calls and feeding the results back, until the LLM
    produces a final response with no tool calls or MAX_TOOL_ROUNDS
    is reached.

    Uses ``current_task_id`` from the subgraph state to identify the
    task to work on.

    Args:
        state: Current domain subgraph state.

    Returns:
        Partial state update with code_base changes.
    """
    domain = state.get("domain", "")
    task_id = state.get("current_task_id", "")
    task_list = state.get("task_list", [])

    task_desc = ""
    for t in task_list:
        if t["id"] == task_id:
            task_desc = t["description"]
            break

    if not task_id or not task_desc:
        return {}

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
    )

    system_design = state.get("system_design", "")
    code_base = state.get("code_base", {})
    feedback = state.get("review_feedback", "")

    # Build messages for the ReAct agent
    system_prompt = _EXPERT_SYSTEM_PROMPT_TEMPLATE.format(domain=domain)
    messages: list = [SystemMessage(content=system_prompt)]

    # Compact system_design to save tokens
    design_text = system_design
    if len(design_text) > 4000:
        design_text = design_text[:4000] + '... (truncated)'

    user_content = (
        f"## System Architecture\n{design_text}\n\n"
        f"## Your Task\n- **Task ID**: {task_id}\n"
        f"- **Description**: {task_desc}\n\n"
        f"## Existing Code Base Files\n"
    )
    if code_base:
        for fp in sorted(code_base.keys()):
            user_content += f"- `{fp}`\n"
    else:
        user_content += "(empty — this is the first task)\n"

    if feedback:
        user_content += (
            f"\n## Reviewer Feedback (fix these issues)\n{feedback}\n"
        )

    messages.append(HumanMessage(content=user_content))

    # ── ReAct loop: LLM ↔ tool calls ──
    llm_with_tools = llm.bind_tools(EXPERT_TOOLS)
    modified_files: dict[str, str] = {}
    tool_log: list[str] = []
    response = None

    for round_idx in range(MAX_TOOL_ROUNDS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)  # Add assistant response to history

        # If no tool calls, the LLM is done
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        # Execute each tool call and feed results back as ToolMessages
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc.get("id", f"{tool_name}_{round_idx}")
            tool_log.append(f"[round {round_idx + 1}] {tool_name}({tool_args})")

            tool_fn = _TOOL_MAP.get(tool_name)
            if tool_fn:
                tool_result = tool_fn.invoke(tool_args)
                result_str = str(tool_result)
                tool_log.append(f"  → {result_str[:200]}")

                # Track file modifications
                if tool_name == "write_file" and "path" in tool_args:
                    modified_files[tool_args["path"]] = tool_args.get("content", "")
            else:
                result_str = f"ERROR: Unknown tool '{tool_name}'"
                tool_log.append(f"  → {result_str}")

            # Feed the tool result back to the LLM for the next round
            messages.append(
                ToolMessage(content=result_str, tool_call_id=tool_call_id)
            )
    else:
        # Exhausted MAX_TOOL_ROUNDS — log a warning
        tool_log.append(
            f"(ReAct loop capped at {MAX_TOOL_ROUNDS} rounds)"
        )

    # If no tool calls were ever made, note it
    if not tool_log and response is not None and response.content:
        tool_log.append("(Expert produced inline response, no tool calls made)")

    # Merge modified files into code_base
    updated_code_base = dict(code_base)
    updated_code_base.update(modified_files)

    return {
        "code_base": updated_code_base,
    }
