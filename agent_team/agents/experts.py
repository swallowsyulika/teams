"""
Expert agent nodes — Frontend & Backend workers.

Each expert receives a single sub-task, uses tools (read_file, write_file,
bash) via a ReAct agent loop, then submits the result for review.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent_team.graph.config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from agent_team.schemas.models import ExpertSubmission
from agent_team.schemas.state import GraphState
from agent_team.tools.file_tools import read_file, write_file
from agent_team.tools.bash_tool import bash

EXPERT_TOOLS = [read_file, write_file, bash]

_EXPERT_SYSTEM_PROMPT_TEMPLATE = """\
You are the **{domain} Expert** of a multi-agent software development team.

You receive ONE sub-task at a time.  Your job:
1. Read the task description carefully.
2. Understand the system architecture and existing code base.
3. Use the available tools (read_file, write_file, bash) to implement
   the task completely.
4. After you finish, produce a structured ExpertSubmission containing:
   - task_id: the ID of this sub-task
   - domain: "{domain}"
   - modified_files: a dict of {{filepath: content}} for every file you
     created or modified
   - tool_execution_summary: a brief log of what you did

Be thorough but concise.  Write clean, production-quality code.
If the reviewer previously rejected your work, their feedback is included —
fix the issues they raised.
"""


def _build_expert_node(domain: str):
    """Factory that creates an expert node function for the given domain."""

    def expert_node(state: GraphState) -> dict[str, Any]:
        """Expert node — implements a single sub-task using tools.

        Args:
            state: Current graph state.

        Returns:
            Partial state update with expert_submissions and code_base changes.
        """
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=TEMPERATURE,
        )

        # Identify which task this expert is working on
        active_tasks = state.get("current_active_tasks", {})
        task_id = active_tasks.get(domain, "")

        task_list = state.get("task_list", [])
        task_desc = ""
        for t in task_list:
            if t["id"] == task_id:
                task_desc = t["description"]
                break

        if not task_id or not task_desc:
            # No task assigned — return empty
            return {"expert_submissions": []}

        system_design = state.get("system_design", {})
        code_base = state.get("code_base", {})
        feedback = state.get("review_feedback", "")

        # Build messages for the ReAct agent
        system_prompt = _EXPERT_SYSTEM_PROMPT_TEMPLATE.format(domain=domain)
        messages = [SystemMessage(content=system_prompt)]

        user_content = (
            f"## System Architecture\n```json\n{json.dumps(system_design, indent=2)}\n```\n\n"
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

        # Use the LLM with tool binding for a ReAct-style loop
        llm_with_tools = llm.bind_tools(EXPERT_TOOLS)
        response = llm_with_tools.invoke(messages)

        # Process any tool calls in the response
        modified_files: dict[str, str] = {}
        tool_log: list[str] = []

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_log.append(f"{tool_name}({tool_args})")

                # Find and execute the tool
                tool_fn = {t.name: t for t in EXPERT_TOOLS}.get(tool_name)
                if tool_fn:
                    tool_result = tool_fn.invoke(tool_args)
                    tool_log.append(f"  → {tool_result[:200]}")

                    # Track file modifications
                    if tool_name == "write_file" and "path" in tool_args:
                        modified_files[tool_args["path"]] = tool_args.get("content", "")

        # If no tool calls, try to extract code from the response
        if not modified_files and response.content:
            # The expert may have produced code inline — we still record it
            tool_log.append("(Expert produced inline response, no tool calls made)")

        submission = ExpertSubmission(
            task_id=task_id,
            domain=domain,
            modified_files=modified_files,
            tool_execution_summary="\n".join(tool_log),
        )

        # Merge modified files into code_base
        updated_code_base = dict(code_base)
        updated_code_base.update(modified_files)

        return {
            "expert_submissions": [submission.model_dump()],
            "code_base": updated_code_base,
            "current_actor": "task_reviewer",
        }

    expert_node.__name__ = f"{domain}_expert_node"
    expert_node.__qualname__ = f"{domain}_expert_node"
    return expert_node


# Instantiate the two expert nodes
frontend_expert_node = _build_expert_node("frontend")
backend_expert_node = _build_expert_node("backend")
