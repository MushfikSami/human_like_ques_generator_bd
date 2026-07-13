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
    # programmatic language check (0.0-1.0).
    "min_bengali_ratio": float(os.getenv("GEN_MIN_BENGALI_RATIO", 0.55)),
}
