"""
Configuration constants and settings for the agent team system.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

# ── Guardrails ──────────────────────────────
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ── Graph Configuration ─────────────────────
SKIP_PLANNER: bool = os.getenv("SKIP_PLANNER", "false").lower() == "true"
SKIP_PLAN_REVIEWER: bool = os.getenv("SKIP_PLAN_REVIEWER", "false").lower() == "true"

def get_enabled_experts() -> list[str]:
    experts_str = os.getenv("ENABLED_EXPERTS", "frontend,backend")
    # Clean up and split
    return [e.strip() for e in experts_str.split(",") if e.strip()]

ENABLED_EXPERTS: list[str] = get_enabled_experts()

# ── Workspace ───────────────────────────────
WORKSPACE_PATH: Path = Path(os.getenv("WORKSPACE_PATH", "./workspace")).resolve()
