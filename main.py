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
        "system_design": "",
        "task_list": [],
        "code_base": {},
        "retry_counters": {},
        "current_actor": "planner",
        "review_feedback": "",
        "phase": "planning",
    }

    print("Starting agent workflow...\n")

    # Stream with mode="values" and subgraphs=True to catch real-time execution in subgraphs.
    final_state = dict(initial_state)
    global_tasks = {t["id"]: dict(t) for t in initial_state.get("task_list", [])}

    for chunk in graph.stream(
        initial_state,
        {"recursion_limit": 100},
        stream_mode="values",
        subgraphs=True,
    ):
        # With subgraphs=True, chunk is (namespace, state)
        if isinstance(chunk, tuple) and len(chunk) == 2:
            namespace, state_snapshot = chunk
            # Parent graph has empty namespace
            if not namespace:
                final_state = state_snapshot
        else:
            state_snapshot = chunk
            final_state = state_snapshot

        # Update our global task view with real-time status from subgraphs
        for t in state_snapshot.get("task_list", []):
            global_tasks[t["id"]] = dict(t)

        actor = state_snapshot.get("current_actor", "")
        if not actor:
            actor = f"subgraph:{state_snapshot.get('domain', 'unknown')}"
            
        phase = state_snapshot.get("phase", "execution")
        
        # Calculate progress using our global track
        completed = sum(1 for t in global_tasks.values() if t.get("status") == "completed")
        failed = sum(1 for t in global_tasks.values() if t.get("status") == "failed")
        total = len(global_tasks)

        status_line = ""
        if total > 0:
            status_line = f" | Progress: {completed}/{total} done"
            if failed:
                status_line += f", {failed} failed"

        print(f"  [step] → next={actor} phase={phase}{status_line}")

    print(f"\n{'='*60}")
    print("  EXECUTION COMPLETE")
    print(f"{'='*60}")

    # Report results — final_state is the authoritative accumulated state
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
