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
