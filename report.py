"""
report.py — Run Quality & Coverage Report

Summarises the current state of the database: persona status counts, matrix
coverage (region / profession / pain-point), duplicate rate, judge-fail rate,
non-Bengali rate, and average question length. Invoked via `main.py --report`.
"""

import logging

import psycopg2.extras

import db

logger = logging.getLogger(__name__)


def _fetchall(conn, sql, args=None):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, args or ())
        return cur.fetchall()


def _hist(conn, field, limit=12):
    """Top-N coverage histogram for a persona json_metadata field."""
    sql = f"""
        SELECT json_metadata->>'{field}' AS val, COUNT(*) AS n
        FROM personas
        GROUP BY val ORDER BY n DESC LIMIT %s;
    """
    return _fetchall(conn, sql, (limit,))


def print_report():
    """Print the run report to stdout."""
    db.init_pool()
    conn = db.get_connection()
    try:
        # Persona status
        status = _fetchall(conn,
            "SELECT status, COUNT(*) n FROM personas GROUP BY status ORDER BY n DESC;")

        # Question-level quality
        q = _fetchall(conn, f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE quality_flags->'judge'->>'verdict' = 'FAIL') AS judge_fail,
                COUNT(*) FILTER (WHERE (quality_flags->>'programmatic_pass')::boolean IS FALSE) AS prog_fail,
                COUNT(*) FILTER (WHERE (quality_flags->>'duplicate')::boolean IS TRUE) AS duplicates,
                AVG(LENGTH(question_text)) AS avg_len,
                AVG((quality_flags->>'bengali_ratio')::float) AS avg_bengali
            FROM {db.QUESTIONS_TABLE};
        """)[0]

        # Distinct dedup hashes vs total (dup rate)
        dup = _fetchall(conn, f"""
            SELECT COUNT(*) total, COUNT(DISTINCT dedup_hash) distinct_hashes
            FROM {db.QUESTIONS_TABLE} WHERE dedup_hash IS NOT NULL;
        """)[0]

        # Procedural-memory diversity: distinct-opener ratio
        openers = _fetchall(conn, f"""
            SELECT COUNT(*) total, COUNT(DISTINCT opener) distinct_openers
            FROM {db.QUESTIONS_TABLE} WHERE opener IS NOT NULL;
        """)[0]

        print("\n" + "=" * 60)
        print("  RUN REPORT")
        print("=" * 60)

        print("\nPersona status:")
        for r in status:
            print(f"  {r['status'] or 'NULL':<10} {r['n']:>8}")

        total = q["total"] or 0
        print(f"\nQuestions generated: {total}")
        if total:
            print(f"  judge FAIL:        {q['judge_fail']}  ({100*q['judge_fail']/total:.1f}%)")
            print(f"  programmatic fail: {q['prog_fail']}  ({100*q['prog_fail']/total:.1f}%)")
            print(f"  flagged duplicate: {q['duplicates']}  ({100*q['duplicates']/total:.1f}%)")
            print(f"  avg length:        {(q['avg_len'] or 0):.0f} chars")
            print(f"  avg Bengali ratio: {(q['avg_bengali'] or 0):.2f}")
        if dup["total"]:
            dr = 100 * (1 - dup["distinct_hashes"] / dup["total"])
            print(f"  hash duplicate rate: {dr:.1f}% "
                  f"({dup['distinct_hashes']} distinct / {dup['total']})")
        if openers["total"]:
            odr = 100 * openers["distinct_openers"] / openers["total"]
            print(f"  distinct-opener ratio: {odr:.1f}% "
                  f"({openers['distinct_openers']} distinct / {openers['total']}) "
                  f"[higher = more diverse phrasing]")

        for field in ("location", "profession", "pain_point", "education"):
            print(f"\nTop {field} coverage:")
            for r in _hist(conn, field):
                print(f"  {str(r['val'])[:34]:<34} {r['n']:>6}")

        print("=" * 60 + "\n")
    finally:
        db.put_connection(conn)
        db.close_pool()
