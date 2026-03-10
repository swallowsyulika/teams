"""
CLI entry point for the Multi-Agent Collaboration Development Team.

Usage:
    python main.py
    python main.py --requirement "Build a REST API with health check"
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from agent_team.graph.builder import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Collaboration Development Team"
    )
    parser.add_argument(
        "--requirement", "-r",
        type=str,
        default=None,
        help="Software requirement to process. If omitted, reads from stdin.",
    )
    args = parser.parse_args()

    if args.requirement:
        requirement = args.requirement
    else:
        print("Enter your software requirement (Ctrl+Z / Ctrl+D to finish):")
        requirement = sys.stdin.read().strip()

    if not requirement:
        print("ERROR: No requirement provided.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Multi-Agent Collaboration Development Team")
    print(f"{'='*60}")
    print(f"\nRequirement: {requirement[:200]}{'...' if len(requirement) > 200 else ''}\n")

    # Build and compile the graph
    graph = build_graph()

    # Initial state
    initial_state = {
        "original_requirement": requirement,
        "system_design": {},
        "task_list": [],
        "current_active_tasks": {},
        "code_base": {},
        "retry_counters": {},
        "current_actor": "planner",
        "review_feedback": "",
        "phase": "planning",
        "expert_submissions": [],
    }

    print("Starting agent workflow...\n")

    # Stream execution for visibility
    for event in graph.stream(initial_state, {"recursion_limit": 100}):
        for node_name, node_output in event.items():
            actor = node_output.get("current_actor", "")
            phase = node_output.get("phase", "")
            task_list = node_output.get("task_list", [])

            completed = sum(1 for t in task_list if t.get("status") == "completed")
            failed = sum(1 for t in task_list if t.get("status") == "failed")
            total = len(task_list)

            status_line = ""
            if total > 0:
                status_line = f" | Progress: {completed}/{total} done"
                if failed:
                    status_line += f", {failed} failed"

            print(f"  [{node_name}] → next={actor} phase={phase}{status_line}")

    # Get final state
    final_state = graph.invoke(initial_state, {"recursion_limit": 100})

    print(f"\n{'='*60}")
    print("  EXECUTION COMPLETE")
    print(f"{'='*60}")

    # Report results
    task_list = final_state.get("task_list", [])
    code_base = final_state.get("code_base", {})

    print(f"\nTasks: {len(task_list)}")
    for t in task_list:
        status_icon = {"completed": "✓", "failed": "✗", "pending": "○"}.get(
            t["status"], "?"
        )
        print(f"  {status_icon} [{t['domain']}] {t['id']}: {t['description']}")

    print(f"\nGenerated files: {len(code_base)}")
    for fp in sorted(code_base.keys()):
        print(f"  📄 {fp}")

    if code_base:
        print(f"\nFiles written to workspace. Review the generated code base.")


if __name__ == "__main__":
    main()
