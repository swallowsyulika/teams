# Subgraph-based Task Queue Refactoring

## Problem

LangGraph's `Send` API has a synchronization barrier: the parent graph waits for *all* dispatched branches to complete their super-step before moving forward. If the Leader dispatches one task to Frontend and one to Backend, a fast-finishing Frontend cannot pull its next task until the slower Backend also finishes its execution step.

## Proposed Design: Independent Domain Subgraphs

We will use **LangGraph Subgraphs** to give each domain ([frontend](file:///d:/gemini/antigravity/agent_team/tests/test_schemas.py#21-25) and [backend](file:///d:/gemini/antigravity/agent_team/tests/test_schemas.py#26-29)) its own isolated execution room and task queue.

1. **Dispatcher (Main Graph)**: The main [leader](file:///d:/gemini/antigravity/agent_team/agent_team/agents/leader.py#37-115) node simply takes the global [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93) and uses `Send` to spawn two Subgraphs—one for Frontend, one for Backend.
2. **Domain Subgraph**: A self-contained `StateGraph` that loops through its assigned tasks independently.
   - It contains three nodes: `task_selector`, [expert](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187), and [reviewer](file:///d:/gemini/antigravity/agent_team/agent_team/agents/reviewer.py#61-87).
   - **Internal Loop**: `task_selector` pops the next pending task. The [expert](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187) executes it. The [reviewer](file:///d:/gemini/antigravity/agent_team/agent_team/agents/reviewer.py#61-87) evaluates it. If it passes, the loop goes right back to `task_selector` to grab the next task. This continues until the queue is empty.
3. **True Asynchrony**: Because the loop happens *inside* the subgraph, Frontend can churn through 5 tasks sequentially while Backend is still working on its first task. They never block each other.
4. **State Merging**: When a Subgraph finishes its queue, it returns its final [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93) and `code_base`, which are merged back into the parent's global state using custom reducers.

---

## Proposed Changes

### State Layer

#### [MODIFY] [state.py](file:///d:/gemini/antigravity/agent_team/agent_team/schemas/state.py)
- Modify [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93) to use a custom reducer (`_merge_task_list`) that merges tasks by ID. This ensures subgraph updates (e.g., status changes to [completed](file:///d:/gemini/antigravity/agent_team/tests/test_circuit_breaker.py#118-138) or `failed`) are smoothly applied to the global list without duplicating tasks.
  - **Idempotency Requirement**: The reducer must merge lists such that newer status overrides older status. Implementation: convert both lists to dicts keyed by [id](file:///d:/gemini/antigravity/agent_team/tests/test_schemas.py#42-46), update `old_dict` with `new_dict`, and return [list(merged_dict.values())](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93).
- Remove `current_active_tasks`, `expert_submissions` reducers as they are no longer needed globally.
- Add a new `DomainState` `TypedDict` for the subgraph:
  - [domain](file:///d:/gemini/antigravity/agent_team/tests/test_schemas.py#30-33): str
  - [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93): Annotated[list[dict], _merge_task_list]
  - `code_base`: Annotated[dict, _merge_dicts]
  - `system_design`: str
  - `current_task_id`: str (for the active task in the subgraph)
  - `review_feedback`: str
  - `retry_count`: int

---

### Models Layer

#### [MODIFY] [models.py](file:///d:/gemini/antigravity/agent_team/agent_team/schemas/models.py)
- Remove [LeaderDecision](file:///d:/gemini/antigravity/agent_team/agent_team/schemas/models.py#73-87) and [DispatchedTask](file:///d:/gemini/antigravity/agent_team/agent_team/schemas/models.py#62-71) — the leader is now deterministic and doesn't use an LLM.

---

### Agent Layer

#### [MODIFY] [leader.py](file:///d:/gemini/antigravity/agent_team/agent_team/agents/leader.py)
- Remove the LLM call. The [leader_node](file:///d:/gemini/antigravity/agent_team/agent_team/agents/leader.py#37-115) can just be a passthrough (returning `{}`).
- The actual parallel dispatch happens in the conditional edge router ([route_after_leader](file:///d:/gemini/antigravity/agent_team/agent_team/graph/builder.py#38-62)), which uses `Send` to invoke `frontend_subgraph` and `backend_subgraph`, passing in the relevant state ([domain](file:///d:/gemini/antigravity/agent_team/tests/test_schemas.py#30-33), [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93), `code_base`, `system_design`).

#### [MODIFY] [experts.py](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py)
- Add a new `task_selector_node`: scans [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93) for the first `"pending"` task matching [domain](file:///d:/gemini/antigravity/agent_team/tests/test_schemas.py#30-33).
   - If found: marks it `"in_progress"`, sets `current_task_id`, and resets `retry_count` and `review_feedback`.
   - If not found: sets `current_task_id = ""` (which will route the subgraph to END).
- Modify [expert_node](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187): use `current_task_id` from the subgraph state instead of looking at global `current_active_tasks`. Return updated `code_base`. Remove dependency on `expert_submissions` list.

#### [MODIFY] [reviewer.py](file:///d:/gemini/antigravity/agent_team/agent_team/agents/reviewer.py)
- [_review_plan](file:///d:/gemini/antigravity/agent_team/agent_team/agents/reviewer.py#89-135) stays in the main graph and is unchanged.
- [_review_task](file:///d:/gemini/antigravity/agent_team/agent_team/agents/reviewer.py#137-263) operates within the subgraph:
   - Reads `current_task_id` and `code_base`.
   - Evaluates the code.
   - If passed: marks `current_task_id` as `"completed"` in [task_list](file:///d:/gemini/antigravity/agent_team/tests/conftest.py#84-93).
   - If failed: increments `retry_count`. If hit `MAX_RETRIES`, marks task `"failed"`. Otherwise updates `review_feedback`.

---

### Graph Layer

#### [MODIFY] [builder.py](file:///d:/gemini/antigravity/agent_team/agent_team/graph/builder.py)
- Add `build_domain_subgraph()` function:
   - Contains nodes: `task_selector`, [expert](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187), `task_reviewer`.
   - Edges: `START` -> `task_selector`.
   - Conditional from `task_selector`: if `current_task_id` == "" -> `END`, else -> [expert](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187).
   - [expert](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187) -> `task_reviewer`.
   - Conditional from `task_reviewer`: if `task_status` == "in_progress" (needs retry) -> [expert](file:///d:/gemini/antigravity/agent_team/agent_team/agents/experts.py#55-187), else (passed/failed) -> `task_selector`.
- Main Graph updates:
   - Replace expert/reviewer nodes with the compiled subgraphs: `frontend_subgraph` and `backend_subgraph`.
   - Setup Phase 1 (planner -> plan_reviewer -> dispatcher).
   - `dispatcher` uses conditional edge to `Send` state to `frontend_subgraph` and `backend_subgraph`.

---

### Tests

#### [MODIFY] tests/ ([test_graph.py](file:///d:/gemini/antigravity/agent_team/tests/test_graph.py), [test_circuit_breaker.py](file:///d:/gemini/antigravity/agent_team/tests/test_circuit_breaker.py), [conftest.py](file:///d:/gemini/antigravity/agent_team/tests/conftest.py))
- Update mock initial states to remove `expert_submissions` and `current_active_tasks`.
- Update [test_graph.py](file:///d:/gemini/antigravity/agent_team/tests/test_graph.py) to test both the `build_domain_subgraph()` loop and the main graph.
- Update [test_circuit_breaker.py](file:///d:/gemini/antigravity/agent_team/tests/test_circuit_breaker.py) to evaluate the reviewer inside the subgraph context with its decoupled retry counter.

---

## Verification Plan

Run the automated test suite (`pytest`) to ensure:
1. Subgraphs compile properly and loop termination conditions are correct.
2. The custom `_merge_task_list` reducer perfectly blends changes back to the global state.
3. The circuit breaker logic correctly fails individual tasks but allows the queue loop to continue.
