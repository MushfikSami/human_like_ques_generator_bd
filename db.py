"""
db.py — PostgreSQL Database Helpers

Provides a shared ThreadedConnectionPool, table initialisation + schema
migration, transactional inserts for personas and generated questions, a
status-driven fetch (pending → done/failed) with a dead-letter path, and a
single-writer CSV backup.

All write operations use explicit commit/rollback for data integrity across
long-running 25,000-record generation runs.

Concurrency notes:
  * LLM calls run under asyncio; DB work is dispatched to a thread pool via the
    ThreadedConnectionPool so many coroutines can share a bounded set of
    connections instead of one connection per persona.
  * CSV appends are NOT thread-safe. They must only be called from the single
    writer task in question_generator.py.
"""

import csv
import os
import logging

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from config import DB_CONFIG, GEN_CONFIG

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

# NOTE: this database (gov_spider_db) is shared with another project that owns a
# differently-shaped `generated_questions` table. We use a dedicated table name
# so the two never collide.
QUESTIONS_TABLE = "hlq_questions"

CREATE_QUESTIONS_TABLE = f"""
CREATE TABLE IF NOT EXISTS {QUESTIONS_TABLE} (
    question_id  SERIAL PRIMARY KEY,
    persona_id   INT REFERENCES personas(persona_id),
    question_text TEXT,
    cot_log      TEXT,
    random_seed  INT,
    dedup_hash   TEXT,
    quality_flags JSONB,
    embedding    BYTEA,
    opener       TEXT,
    quality_score REAL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Schema migration — additive columns introduced by the reworked engine.
# Each runs independently and is safe to re-apply.
MIGRATIONS = [
    "ALTER TABLE personas ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending';",
    "ALTER TABLE personas ADD COLUMN IF NOT EXISTS attempts INT DEFAULT 0;",
    "ALTER TABLE personas ADD COLUMN IF NOT EXISTS error TEXT;",
    # Backfill status from the legacy `processed` boolean.
    "UPDATE personas SET status = 'done' WHERE processed = TRUE AND status = 'pending';",
    f"ALTER TABLE {QUESTIONS_TABLE} ADD COLUMN IF NOT EXISTS dedup_hash TEXT;",
    f"ALTER TABLE {QUESTIONS_TABLE} ADD COLUMN IF NOT EXISTS quality_flags JSONB;",
    # Procedural memory columns (see docs/procedural_memory_design.md)
    f"ALTER TABLE {QUESTIONS_TABLE} ADD COLUMN IF NOT EXISTS embedding BYTEA;",
    f"ALTER TABLE {QUESTIONS_TABLE} ADD COLUMN IF NOT EXISTS opener TEXT;",
    f"ALTER TABLE {QUESTIONS_TABLE} ADD COLUMN IF NOT EXISTS quality_score REAL;",
    "CREATE INDEX IF NOT EXISTS idx_personas_status ON personas(status);",
    f"CREATE INDEX IF NOT EXISTS idx_hlq_dedup ON {QUESTIONS_TABLE}(dedup_hash);",
]

INSERT_PERSONA = """
INSERT INTO personas (age, gender, location, profession, social_status, backstory, json_metadata)
VALUES (%(age)s, %(gender)s, %(location)s, %(profession)s, %(social_status)s, %(backstory)s, %(json_metadata)s)
RETURNING persona_id;
"""

# Atomically CLAIM a batch: flip pending → processing and return the rows in one
# statement. This is what prevents a persona being fetched (and generated) twice
# while an earlier result is still in flight — no cross-batch barrier needed.
# FOR UPDATE SKIP LOCKED makes it safe under concurrency.
FETCH_PENDING = """
WITH claimed AS (
    SELECT persona_id FROM personas
    WHERE status = 'pending'
    ORDER BY persona_id
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
UPDATE personas p SET status = 'processing'
FROM claimed
WHERE p.persona_id = claimed.persona_id
RETURNING p.persona_id, p.age, p.gender, p.location, p.profession,
          p.social_status, p.backstory, p.json_metadata, p.attempts;
"""

# Reset any rows stranded in 'processing' by a previous crash back to 'pending'.
RESET_PROCESSING = "UPDATE personas SET status = 'pending' WHERE status = 'processing';"

INSERT_QUESTION = f"""
INSERT INTO {QUESTIONS_TABLE}
    (persona_id, question_text, cot_log, random_seed, dedup_hash, quality_flags,
     embedding, opener, quality_score)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

MARK_DONE = "UPDATE personas SET status = 'done', processed = TRUE WHERE persona_id = %s;"
MARK_FAILED = "UPDATE personas SET status = 'failed', error = %s WHERE persona_id = %s;"
# On retry, bump the counter AND release the claim (processing → pending) so the
# persona re-enters the pool for another attempt.
INCREMENT_ATTEMPT = ("UPDATE personas SET attempts = attempts + 1, status = 'pending' "
                     "WHERE persona_id = %s;")


# ─── Connection Pool ─────────────────────────────────────────────────────────

_POOL: ThreadedConnectionPool | None = None


def init_pool():
    """Create the global ThreadedConnectionPool (idempotent)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadedConnectionPool(
            GEN_CONFIG["pool_min"], GEN_CONFIG["pool_max"], **DB_CONFIG
        )
        logger.info("Initialised DB connection pool (min=%d, max=%d) for '%s'",
                    GEN_CONFIG["pool_min"], GEN_CONFIG["pool_max"], DB_CONFIG["dbname"])
    return _POOL


def close_pool():
    """Close all pooled connections."""
    global _POOL
    if _POOL is not None:
        _POOL.closeall()
        _POOL = None
        logger.info("Closed DB connection pool.")


def get_connection():
    """
    Return a connection.

    If the pool is initialised, borrow from it; callers MUST return it via
    put_connection(). If not, fall back to a standalone connection (used by
    one-shot CLI actions like --init-db and --gen-personas).
    """
    if _POOL is not None:
        return _POOL.getconn()
    return psycopg2.connect(**DB_CONFIG)


def put_connection(conn):
    """Return a connection to the pool, or close it if pooling is disabled."""
    if _POOL is not None:
        _POOL.putconn(conn)
    else:
        conn.close()


# ─── Table Initialisation & Migration ────────────────────────────────────────

def init_tables():
    """Create base tables (if absent) and initialise the CSV backup."""
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
        put_connection(conn)

    _init_csv()


def migrate():
    """Apply additive schema migrations (status/attempts/error/dedup columns)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for stmt in MIGRATIONS:
                cur.execute(stmt)
        conn.commit()
        logger.info("Applied %d schema migrations.", len(MIGRATIONS))
    except Exception:
        conn.rollback()
        logger.exception("Failed to apply schema migrations.")
        raise
    finally:
        put_connection(conn)


def _init_csv():
    """Create the CSV backup file with headers if it doesn't already exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        logger.info("Initialised CSV backup at %s", CSV_PATH)


# ─── Persona Operations ─────────────────────────────────────────────────────

def insert_persona(conn, persona: dict) -> int:
    """Transactionally insert a single persona; returns the new persona_id."""
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


def bulk_insert_personas(conn, personas: list[dict]):
    """
    Bulk-insert personas using execute_values for speed at 25k scale.

    Args:
        conn: Active connection.
        personas: List of persona dicts (same keys as insert_persona).
    """
    rows = [
        (p["age"], p["gender"], p["location"], p["profession"],
         p["social_status"], p["backstory"], p["json_metadata"])
        for p in personas
    ]
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO personas (age, gender, location, profession, "
                "social_status, backstory, json_metadata) VALUES %s",
                rows,
                page_size=500,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Bulk persona insert failed (%d rows).", len(rows))
        raise


def fetch_memory_rows(conn) -> list[dict]:
    """
    Load rows needed to prime procedural memory: embedding + opener + score,
    joined to the persona's profession/region/topic. Only rows that have an
    embedding are returned (older rows without one are skipped).
    """
    sql = f"""
        SELECT q.embedding, q.opener, q.quality_score, q.question_text,
               p.json_metadata->>'profession' AS profession,
               p.json_metadata->>'location'   AS region,
               p.json_metadata->>'pain_point' AS topic
        FROM {QUESTIONS_TABLE} q
        JOIN personas p USING (persona_id)
        WHERE q.embedding IS NOT NULL;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


def count_personas(conn, status: str = None) -> int:
    """Count personas, optionally filtered by status."""
    with conn.cursor() as cur:
        if status is None:
            cur.execute("SELECT COUNT(*) FROM personas;")
        else:
            cur.execute("SELECT COUNT(*) FROM personas WHERE status = %s;", (status,))
        return cur.fetchone()[0]


def fetch_pending(conn, batch_size: int) -> list[dict]:
    """
    Atomically CLAIM up to `batch_size` pending personas (pending → processing)
    and return them. Committed immediately so the claim is durable.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(FETCH_PENDING, (batch_size,))
        rows = cur.fetchall()
    conn.commit()
    return [dict(row) for row in rows]


def reset_processing(conn) -> int:
    """Release rows stranded in 'processing' (e.g. after a crash) back to pending."""
    with conn.cursor() as cur:
        cur.execute(RESET_PROCESSING)
        n = cur.rowcount
    conn.commit()
    return n


def increment_attempt(conn, persona_id: int):
    """Increment the retry counter for a persona (committed immediately)."""
    try:
        with conn.cursor() as cur:
            cur.execute(INCREMENT_ATTEMPT, (persona_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def mark_failed(conn, persona_id: int, error: str):
    """Mark a persona as permanently failed (dead-letter)."""
    try:
        with conn.cursor() as cur:
            cur.execute(MARK_FAILED, (error[:2000], persona_id))
        conn.commit()
        logger.warning("persona_id=%d marked FAILED: %s", persona_id, error[:200])
    except Exception:
        conn.rollback()
        raise


# ─── Question Operations ────────────────────────────────────────────────────

def save_question(conn, persona_id: int, question_text: str, cot_log: str,
                  random_seed: int, dedup_hash: str = None,
                  quality_flags: dict = None, embedding: bytes = None,
                  opener: str = None, quality_score: float = None):
    """
    Transactionally insert a generated question and mark the persona done.

    The question insert and the persona status update commit atomically.
    `embedding`/`opener`/`quality_score` back the procedural-memory subsystem
    and may be None when memory is disabled.
    """
    flags_json = psycopg2.extras.Json(quality_flags) if quality_flags is not None else None
    emb = psycopg2.Binary(embedding) if embedding is not None else None
    try:
        with conn.cursor() as cur:
            cur.execute(INSERT_QUESTION,
                        (persona_id, question_text, cot_log, random_seed,
                         dedup_hash, flags_json, emb, opener, quality_score))
            cur.execute(MARK_DONE, (persona_id,))
        conn.commit()
        logger.info("Saved question for persona_id=%d", persona_id)
    except Exception:
        conn.rollback()
        logger.exception("Failed to save question for persona_id=%d", persona_id)
        raise


# ─── CSV Backup (single-writer only) ─────────────────────────────────────────

def append_to_csv(row: dict):
    """
    Append a single row to the CSV backup.

    NOT thread-safe — call only from the dedicated writer task.
    """
    _init_csv()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
        writer.writerow(row)
