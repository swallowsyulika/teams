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
from pydantic import ValidationError

load_dotenv()

from agent_team.graph.builder import build_graph
from agent_team.graph.config import ENABLED_EXPERTS, SKIP_PLANNER, SKIP_PLAN_REVIEWER, WORKSPACE_PATH
from agent_team.schemas.models import TaskItem


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
    parser.add_argument(
        "--task-file", "-t",
        type=str,
        default=None,
        help="Path to a JSON file containing the task list (required if SKIP_PLANNER is true).",
    )
    args = parser.parse_args()

    if SKIP_PLANNER and not args.task_file:
        print("ERROR: SKIP_PLANNER is true, but no --task-file was provided.")
        sys.exit(1)

    requirement = args.requirement
    
    if not requirement and not args.task_file:
        print("Enter your software requirement (Ctrl+Z / Ctrl+D to finish):")
        requirement = sys.stdin.read().strip()

    if not requirement and not args.task_file:
        print("ERROR: No requirement provided.")
        sys.exit(1)

    requirement = requirement or "Requirement provided via task list file."

    task_list = []
    if args.task_file:
        try:
            with open(args.task_file, "r", encoding="utf-8") as f:
                raw_tasks = json.load(f)
            
            if not isinstance(raw_tasks, list):
                print(f"ERROR: Task file {args.task_file} must contain a JSON array of tasks.")
                sys.exit(1)

            # Validate tasks
            for item in raw_tasks:
                if not isinstance(item, dict):
                    print(f"ERROR: Task file {args.task_file} has invalid structure. Elements must be objects.")
                    sys.exit(1)
                if item.get("domain") not in ENABLED_EXPERTS:
                    print(f"ERROR: Task {item.get('id')} has invalid domain '{item.get('domain')}'. Allowed: {ENABLED_EXPERTS}")
                    sys.exit(1)
                # Validation using Pydantic model
                valid_task = TaskItem(**item)
                task_list.append(valid_task.model_dump())
        except FileNotFoundError:
            print(f"ERROR: Task file {args.task_file} not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERROR: Task file {args.task_file} is not valid JSON.")
            sys.exit(1)
        except ValidationError as e:
            print(f"ERROR: Task validation failed:\n{e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  Multi-Agent Collaboration Development Team")
    print(f"{'='*60}")
    print(f"\nRequirement: {requirement[:200]}{'...' if len(requirement) > 200 else ''}\n")

    # Build and compile the graph
    graph = build_graph()
    
    if SKIP_PLANNER and SKIP_PLAN_REVIEWER:
        starting_actor = "leader"
        starting_phase = "execution"
    elif SKIP_PLANNER and not SKIP_PLAN_REVIEWER:
        starting_actor = "plan_reviewer"
        starting_phase = "planning"
    else:
        starting_actor = "planner"
        starting_phase = "planning"

    # Initial state
    initial_state = {
        "original_requirement": requirement,
        "system_design": "System design bypassed." if SKIP_PLANNER else "",
        "task_list": task_list,
        "code_base": {},
        "retry_counters": {},
        "current_actor": starting_actor,
        "review_feedback": "",
        "phase": starting_phase,
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

        # Export current task list to a JSON file for real-time monitoring
        try:
            WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
            tasks_out_path = WORKSPACE_PATH / "tasks_status.json"
            with open(tasks_out_path, "w", encoding="utf-8") as f:
                json.dump(list(global_tasks.values()), f, indent=2, ensure_ascii=False)
        except Exception:
            pass

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
