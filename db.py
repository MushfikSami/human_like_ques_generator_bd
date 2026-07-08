"""
db.py — PostgreSQL Database Helpers

Provides connection management, table initialisation, transactional inserts
for personas and generated questions, and CSV backup appending.

All write operations use explicit BEGIN/COMMIT/ROLLBACK for data integrity
across long-running 25,000-record generation runs.
"""

import csv
import os
import logging
from datetime import datetime

import psycopg2
import psycopg2.extras

from config import DB_CONFIG

logger = logging.getLogger(__name__)

# Path to the local CSV backup file
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personas_questions.csv")

# CSV column headers matching the combined persona + question schema
CSV_HEADERS = [
    "persona_id", "age", "gender", "location", "profession",
    "social_status", "backstory", "question_text", "cot_log",
    "random_seed", "created_at",
]

# ─── SQL Statements ──────────────────────────────────────────────────────────

CREATE_PERSONAS_TABLE = """
CREATE TABLE IF NOT EXISTS personas (
    persona_id   SERIAL PRIMARY KEY,
    age          INT,
    gender       VARCHAR(50),
    location     VARCHAR(100),
    profession   VARCHAR(100),
    social_status VARCHAR(50),
    backstory    TEXT,
    json_metadata JSONB,
    processed    BOOLEAN DEFAULT FALSE
);
"""

CREATE_QUESTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS generated_questions (
    question_id  SERIAL PRIMARY KEY,
    persona_id   INT REFERENCES personas(persona_id),
    question_text TEXT,
    cot_log      TEXT,
    random_seed  INT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_PERSONA = """
INSERT INTO personas (age, gender, location, profession, social_status, backstory, json_metadata)
VALUES (%(age)s, %(gender)s, %(location)s, %(profession)s, %(social_status)s, %(backstory)s, %(json_metadata)s)
RETURNING persona_id;
"""

FETCH_UNPROCESSED = """
SELECT persona_id, age, gender, location, profession, social_status, backstory, json_metadata
FROM personas
WHERE processed = FALSE
ORDER BY persona_id
LIMIT %s;
"""

INSERT_QUESTION = """
INSERT INTO generated_questions (persona_id, question_text, cot_log, random_seed)
VALUES (%s, %s, %s, %s);
"""

MARK_PROCESSED = """
UPDATE personas SET processed = TRUE WHERE persona_id = %s;
"""


# ─── Connection ──────────────────────────────────────────────────────────────

def get_connection():
    """Return a new psycopg2 connection using DB_CONFIG."""
    conn = psycopg2.connect(**DB_CONFIG)
    logger.info("Connected to PostgreSQL database '%s'", DB_CONFIG["dbname"])
    return conn


# ─── Table Initialisation ───────────────────────────────────────────────────

def init_tables():
    """
    Create the personas and generated_questions tables if they don't exist.
    Also initialises the CSV backup file with headers if it doesn't exist.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_PERSONAS_TABLE)
            cur.execute(CREATE_QUESTIONS_TABLE)
        conn.commit()
        logger.info("Database tables initialised successfully.")
    except Exception:
        conn.rollback()
        logger.exception("Failed to initialise database tables.")
        raise
    finally:
        conn.close()

    # Initialise CSV with headers if it doesn't exist
    _init_csv()


def _init_csv():
    """Create the CSV backup file with headers if it doesn't already exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        logger.info("Initialised CSV backup at %s", CSV_PATH)


# ─── Persona Operations ─────────────────────────────────────────────────────

def insert_persona(conn, persona: dict) -> int:
    """
    Transactionally insert a single persona into the personas table.

    Args:
        conn: Active psycopg2 connection.
        persona: Dict with keys: age, gender, location, profession,
                 social_status, backstory, json_metadata.

    Returns:
        The generated persona_id.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(INSERT_PERSONA, persona)
            persona_id = cur.fetchone()[0]
        conn.commit()
        return persona_id
    except Exception:
        conn.rollback()
        logger.exception("Failed to insert persona: %s", persona.get("profession", "unknown"))
        raise


def fetch_unprocessed_personas(conn, batch_size: int) -> list[dict]:
    """
    Fetch a batch of unprocessed personas from the database.

    Args:
        conn: Active psycopg2 connection.
        batch_size: Maximum number of personas to fetch.

    Returns:
        List of persona dicts with keys matching the table columns.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(FETCH_UNPROCESSED, (batch_size,))
        rows = cur.fetchall()
    return [dict(row) for row in rows]


# ─── Question Operations ────────────────────────────────────────────────────

def save_question(conn, persona_id: int, question_text: str, cot_log: str, random_seed: int):
    """
    Transactionally insert a generated question and mark the persona as processed.

    Uses BEGIN/COMMIT/ROLLBACK to ensure both the question insert and the
    persona status update happen atomically.

    Args:
        conn: Active psycopg2 connection.
        persona_id: The persona this question was generated for.
        question_text: The final generated question text.
        cot_log: The Chain-of-Thought reflection log.
        random_seed: The random seed used for generation.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(INSERT_QUESTION, (persona_id, question_text, cot_log, random_seed))
            cur.execute(MARK_PROCESSED, (persona_id,))
        conn.commit()
        logger.info("Saved question for persona_id=%d", persona_id)
    except Exception:
        conn.rollback()
        logger.exception("Failed to save question for persona_id=%d", persona_id)
        raise


# ─── CSV Backup ──────────────────────────────────────────────────────────────

def append_to_csv(row: dict):
    """
    Append a single row to the personas_questions.csv backup file.

    Args:
        row: Dict with keys matching CSV_HEADERS. Missing keys default to ''.
    """
    _init_csv()  # Ensure the file exists with headers
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writerow(row)
