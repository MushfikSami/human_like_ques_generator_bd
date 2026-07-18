"""
config.py — Centralised Configuration

Loads environment variables from .env and exposes DB_CONFIG and LLM_CONFIG
dictionaries used throughout the application.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL connection parameters
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME", "gov_spider_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
}

# vLLM / OpenAI-compatible endpoint configuration
LLM_CONFIG = {
    "url":   os.getenv("LLM_URL", "http://localhost:5000/v1/chat/completions"),
    "model": os.getenv("LLM_MODEL", "qwen3-35b-awq"),
}

# Generation / batch-engine tuning
GEN_CONFIG = {
    # Concurrent in-flight LLM requests (asyncio semaphore limit).
    "concurrency":  int(os.getenv("GEN_CONCURRENCY", 50)),
    # psycopg2 ThreadedConnectionPool sizing.
    "pool_min":     int(os.getenv("DB_POOL_MIN", 1)),
    "pool_max":     int(os.getenv("DB_POOL_MAX", 20)),
    # Dead-letter: how many times a persona is retried before status='failed'.
    "max_attempts": int(os.getenv("GEN_MAX_ATTEMPTS", 3)),
    # Sampling temperature for the generator call.
    "temperature":  float(os.getenv("GEN_TEMPERATURE", 0.8)),
    # Minimum fraction of Bengali-script characters for a question to pass the
    # programmatic language check (0.0-1.0). Raised to reject transliterated
    # ("romanized") Bengali more aggressively.
    "min_bengali_ratio": float(os.getenv("GEN_MIN_BENGALI_RATIO", 0.65)),
    # Max in-cycle rewrite attempts before the best candidate so far is kept.
    "max_rewrites": int(os.getenv("GEN_MAX_REWRITES", 3)),
}

# Procedural memory (style + exemplar) — see docs/procedural_memory_design.md
MEMORY_CONFIG = {
    "enabled":       os.getenv("MEMORY_ENABLED", "true").lower() == "true",
    # Multilingual embedder (Bengali-capable). Runs on the configured device.
    "model_name":    os.getenv("MEMORY_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
    "device":        os.getenv("MEMORY_DEVICE", "cuda"),   # "cuda" | "cpu"
    # Cosine similarity above which a draft counts as a near-duplicate.
    "sim_threshold": float(os.getenv("MEMORY_SIM_THRESHOLD", 0.92)),
    # How many overused openers to warn the model against.
    "avoid_openers_n": int(os.getenv("MEMORY_AVOID_OPENERS_N", 5)),
    # How many few-shot exemplars to inject.
    "k_exemplars":   int(os.getenv("MEMORY_K_EXEMPLARS", 3)),
    # Minimum quality_score for a question to enter the exemplar bank.
    "min_exemplar_score": float(os.getenv("MEMORY_MIN_EXEMPLAR_SCORE", 1.4)),
    # Number of tokens that make up an "opener" fingerprint.
    "opener_tokens": int(os.getenv("MEMORY_OPENER_TOKENS", 6)),
}
