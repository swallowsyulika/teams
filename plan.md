這就為你奉上這份完美收斂的 **AI Agent Team 協作系統 URD**。這份文件已經將我們的討論精華結構化，可以直接作為你下一步撰寫 LangGraph 程式碼的架構藍圖。

---

# 系統需求規格書 (URD)：多智能體協作開發團隊 (Multi-Agent Dev Team)

## 1. 產品概述 (Product Overview)

本系統旨在利用 LangGraph 與 LangChain 框架，建構一個具備高度擴充性的多智能體 (Multi-Agent) 自動化軟體開發團隊。系統透過 Supervisor (Leader) 與 Worker (Experts) 的架構，實現需求分析、任務拆解、前後端異步開發，以及嚴格的自動化代碼審查，最終交付符合使用者預期的軟體專案。

## 2. 系統架構與工作流程 (Architecture & Workflow)

系統執行分為兩個主要階段，以確保架構穩定與開發並行：

* **Phase 1: 系統初始化 (Initialization)**
1. 使用者輸入原始需求。
2. **Planner** 進行系統架構設計與任務拆解。
3. **Reviewer** 審核 Planner 的計畫，通過後進入 Phase 2；不通過則退回重擬。


* **Phase 2: 並行開發 (Asynchronous Parallel Execution)**
1. **Leader** 讀取全局狀態，將任務清單拆分為最小單位的「小任務」，並獨立分派給各領域專家（如 Frontend Task 1, Backend Task 1）。
2. **Experts (前端/後端)** 在沙盒環境中異步並行作業，互不等待。
3. 專家完成小任務後，直接提交給 **Reviewer** 進行驗證。
4. **Reviewer** 審核通過：更新狀態為完成，路由回 Leader 進行下一個小任務的派發。
5. **Reviewer** 審核失敗：附帶修改建議，直接退回給原 Expert 進行修正（觸發局部迴圈）。



## 3. 角色定義與職責 (Agent Roles & Personas)

| 角色 (Agent) | 核心職責 | 行為準則與限制 |
| --- | --- | --- |
| **Planner** | 系統規劃與任務拆解 | 負責產出系統架構與前後端任務清單。任務粒度必須極小，確保單次生成即可完成。 |
| **Leader** | 任務分派與進度控管 | 不負責寫 code。僅負責讀取 State，判斷當前進度，並每次「只」派發一個小任務給對應的 Expert。 |
| **Expert (Frontend)** | 前端代碼實作 | 接收單一小任務，利用工具讀寫代碼並測試。完成後提交給 Reviewer。 |
| **Expert (Backend)** | 後端代碼實作 | 接收單一小任務，利用工具讀寫代碼並測試。完成後提交給 Reviewer。 |
| **Reviewer** | 嚴格把關與代碼審查 | 檢查邏輯錯誤、惡意代碼及任務完成度。絕不妥協，失敗即退回並給予精準反饋。 |

*(註：Expert 節點採介面化設計，未來可無縫擴充 Database Expert 或 DevOps Expert 等角色)*

## 4. 全局狀態管理 (Global Graph State)

為避免無窮迴圈與狀態混亂，所有 Agent 皆為無狀態 (Stateless)，嚴格依賴以下結構化的 Graph State 進行流轉：

* `original_requirement` (String): 使用者的原始需求。
* `system_design` (Dict): Planner 產出的系統設定與架構。
* `task_list` (List[Dict]): 全局任務清單，包含每個小任務的 ID、描述、所屬領域與當前狀態 (`pending`, `in_progress`, `completed`)。
* `current_active_tasks` (Dict): 記錄各領域當前正在執行的小任務 ID (例如：`{"frontend": "task_2", "backend": "task_1"}`)。
* `code_base` (Dict): 當前累積的程式碼或檔案結構狀態。
* `retry_counters` (Dict): 追蹤每個小任務被 Reviewer 退回的次數 (例如：`{"task_1": 3}`)。
* `current_actor` (String): 狀態機路由指標，標記下一步該喚醒的 Agent。

## 5. 通訊協定與結構化輸出 (Communication & Structured Outputs)

各節點間的資料傳遞強制使用 Pydantic (或 JSON Schema) 進行結構化輸出約束：

* **PlannerOutput**: 包含 `system_architecture`, `frontend_tasks`, `backend_tasks`。
* **LeaderDecision**: 包含 `next_actor`, `dispatched_task_id`。
* **ExpertSubmission**: 包含 `task_id`, `modified_files`, `tool_execution_summary`。
* **ReviewerEvaluation**: 包含 `is_passed` (Boolean), `feedback` (String)。

## 6. 工具與環境設定 (Tools & Environment)

賦予 Expert 執行任務所需的基礎 OS 能力，所有工具必須在安全的環境下運行：

* `read_file`: 讀取特定路徑的檔案內容。
* `write_file`: 寫入或覆蓋檔案代碼。
* `bash`: 執行終端機指令（如編譯、跑測試、安裝依賴套件）。
* **安全規範**：包含 `bash` 在內的所有工具執行，必須被限制在獨立的 Docker Container 或 Sandbox (沙盒) 環境中，避免破壞本機系統。

## 7. 系統護欄與邊界條件 (Guardrails & Edge Cases)

* **斷路器機制 (Circuit Breaker)**：系統全域設定 `max_retries = 10`。若單一小任務的退回次數達到此上限，Reviewer 將攔截該任務，標記為失敗並暫停該分支的工作流，等待人類介入或由 Leader 重新評估，徹底防止無限迴圈。

---

這份 URD 已經將你的構想轉化為具備實作可行性的工程藍圖。

**下一步，需要我幫你直接生成這個 LangGraph 的 `State` 定義代碼 (TypedDict / Pydantic)，或是先建構各個 Agent 的 System Prompts 嗎？**