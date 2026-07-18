"""
judge_config.py — configuration for the standalone two-axis judge.

Reuses the parent project's LLM_CONFIG / DB_CONFIG (same vLLM endpoint and
database); adds judge-specific tuning read from the environment.
"""

import os
import sys

# Make the parent project importable (config.py, db.py, cot_module.py).
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from config import LLM_CONFIG, DB_CONFIG  # noqa: E402  (re-exported for judge use)

JUDGE_CONFIG = {
    # Concurrent in-flight judge requests to vLLM.
    "concurrency": int(os.getenv("JUDGE_CONCURRENCY", 25)),
    # Low temperature → consistent, repeatable verdicts.
    "temperature": float(os.getenv("JUDGE_TEMPERATURE", 0.1)),
    # Enough for two short reasons + JSON envelope.
    "max_tokens": int(os.getenv("JUDGE_MAX_TOKENS", 500)),
    # qwen3 "thinking": off by default for speed. Turn on (JUDGE_THINKING=true)
    # for deeper reasoning when vLLM is idle (generation finished).
    "thinking": os.getenv("JUDGE_THINKING", "false").lower() == "true",
    # Request timeout (seconds).
    "timeout": int(os.getenv("JUDGE_TIMEOUT", 120)),
}
