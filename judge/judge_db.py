"""
judge_db.py — persistence for the two-axis judge.

Owns the `judge_evaluations` table (verdicts + reasons per axis), a resumable
fetch of not-yet-judged questions, a save, and a CSV mirror. Reuses the parent
project's connection pool from db.py.
"""

import csv
import os

import judge_config  # noqa: F401  (sets sys.path so `db` is importable)
import db
import psycopg2.extras

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_evaluations.csv")

CSV_HEADERS = [
    "question_id", "persona_id", "profession", "location", "education",
    "pain_point", "question_text", "axis1_pass", "axis1_reason",
    "axis2_pass", "axis2_reason", "overall_pass",
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS judge_evaluations (
    eval_id      SERIAL PRIMARY KEY,
    question_id  INT UNIQUE REFERENCES hlq_questions(question_id),
    persona_id   INT,
    axis1_pass   BOOLEAN,
    axis1_reason TEXT,
    axis2_pass   BOOLEAN,
    axis2_reason TEXT,
    overall_pass BOOLEAN,
    raw_response TEXT,
    model        VARCHAR(100),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Fetch questions that have not been judged yet, flattening the persona profile
# out of json_metadata so the prompt builder gets plain fields.
FETCH_UNJUDGED = f"""
SELECT q.question_id, q.persona_id, q.question_text,
       (p.json_metadata->>'age')::int      AS age,
       p.json_metadata->>'gender'          AS gender,
       p.json_metadata->>'location'        AS location,
       p.json_metadata->>'profession'      AS profession,
       p.json_metadata->>'social_status'   AS social_status,
       p.json_metadata->>'education'        AS education,
       p.json_metadata->>'pain_point'       AS pain_point,
       p.backstory                          AS backstory
FROM {db.QUESTIONS_TABLE} q
JOIN personas p USING (persona_id)
WHERE NOT EXISTS (
    SELECT 1 FROM judge_evaluations e WHERE e.question_id = q.question_id
)
ORDER BY q.question_id
LIMIT %s;
"""

INSERT_EVAL = """
INSERT INTO judge_evaluations
    (question_id, persona_id, axis1_pass, axis1_reason, axis2_pass, axis2_reason,
     overall_pass, raw_response, model)
VALUES (%(question_id)s, %(persona_id)s, %(axis1_pass)s, %(axis1_reason)s,
        %(axis2_pass)s, %(axis2_reason)s, %(overall_pass)s, %(raw_response)s, %(model)s)
ON CONFLICT (question_id) DO NOTHING;
"""


def init():
    """Create the judge_evaluations table and the CSV header if absent."""
    db.init_pool()
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE)
        conn.commit()
    finally:
        db.put_connection(conn)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)


def fetch_unjudged(conn, limit: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(FETCH_UNJUDGED, (limit,))
        return [dict(r) for r in cur.fetchall()]


def count_unjudged(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT COUNT(*) FROM {db.QUESTIONS_TABLE} q
            WHERE NOT EXISTS (SELECT 1 FROM judge_evaluations e
                              WHERE e.question_id = q.question_id);
        """)
        return cur.fetchone()[0]


def save_eval(conn, row: dict):
    """Insert one evaluation (idempotent on question_id)."""
    try:
        with conn.cursor() as cur:
            cur.execute(INSERT_EVAL, row)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def append_csv(row: dict):
    """Append one evaluation to the CSV mirror (single-writer only)."""
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore").writerow(row)
