"""
run_judge.py — standalone two-axis LLM judge over generated questions.

Judges every row in `hlq_questions` against Axis 1 (linguistic/socio-demographic
alignment) and Axis 2 (contextual/pragmatic correctness). Resumable: already
judged questions are skipped. Verdicts + per-axis reasons go to the
`judge_evaluations` table and `judge/judge_evaluations.csv`.

Usage:
    python judge/run_judge.py                 # judge all unjudged (resumable)
    python judge/run_judge.py --limit 500     # judge up to 500 unjudged
    python judge/run_judge.py --batch-size 25 # concurrency
    python judge/run_judge.py --report        # print pass-rate summary
"""

import argparse
import asyncio
import json
import logging
import re
import sys

import aiohttp

import judge_config
from judge_config import JUDGE_CONFIG, LLM_CONFIG
import judge_prompt
import judge_db
import db
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("judge")

# ─── Robust JSON verdict parsing (adapted from cot_module.parse_judge) ────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_brace_objects(text: str):
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _axis_from(obj, key):
    a = obj.get(key) or {}
    if isinstance(a, dict):
        verdict = str(a.get("verdict", "")).upper()
        reason = (a.get("reason") or "").strip()
    else:
        verdict, reason = str(a).upper(), ""
    return ("PASS" if verdict == "PASS" else ("FAIL" if verdict == "FAIL" else None)), reason


def parse_verdict(raw: str) -> dict | None:
    """
    Parse the judge JSON into {axis1_pass, axis1_reason, axis2_pass, axis2_reason,
    overall_pass}. Returns None if unparseable, or if an axis is FAIL without a
    reason (so it is re-judged rather than stored as a bare FAIL).
    """
    text = _THINK_RE.sub("", raw or "").strip()
    fence = _FENCE_RE.search(text)
    scan = fence.group(1) if fence else text

    obj = None
    for cand in _iter_brace_objects(scan):
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and ("axis1" in data or "axis2" in data):
                obj = data
                break
        except json.JSONDecodeError:
            continue
    if obj is None:
        return None

    a1, r1 = _axis_from(obj, "axis1")
    a2, r2 = _axis_from(obj, "axis2")
    if a1 is None or a2 is None:
        return None
    # A FAIL with no concrete reason is treated as a parse failure.
    if (a1 == "FAIL" and not r1) or (a2 == "FAIL" and not r2):
        return None

    a1_pass, a2_pass = (a1 == "PASS"), (a2 == "PASS")
    return {
        "axis1_pass": a1_pass, "axis1_reason": r1,
        "axis2_pass": a2_pass, "axis2_reason": r2,
        "overall_pass": a1_pass and a2_pass,   # recomputed, not trusted
    }


# ─── LLM call ────────────────────────────────────────────────────────────────

async def _call_llm(session, messages):
    payload = {
        "model": LLM_CONFIG["model"],
        "messages": messages,
        "temperature": JUDGE_CONFIG["temperature"],
        "max_tokens": JUDGE_CONFIG["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": JUDGE_CONFIG["thinking"]},
    }
    for attempt in range(3):
        try:
            async with session.post(
                LLM_CONFIG["url"], json=payload,
                timeout=aiohttp.ClientTimeout(total=JUDGE_CONFIG["timeout"]),
            ) as resp:
                if resp.status != 200:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(f"LLM {resp.status}: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.warning("LLM call failed (%d/3): %s", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise


# ─── Per-question evaluation ─────────────────────────────────────────────────

def _db_save(row, persona, question_text):
    conn = db.get_connection()
    try:
        judge_db.save_eval(conn, row)
    finally:
        db.put_connection(conn)
    judge_db.append_csv({
        "question_id": row["question_id"], "persona_id": row["persona_id"],
        "profession": persona.get("profession"), "location": persona.get("location"),
        "education": persona.get("education"), "pain_point": persona.get("pain_point"),
        "question_text": question_text,
        "axis1_pass": row["axis1_pass"], "axis1_reason": row["axis1_reason"],
        "axis2_pass": row["axis2_pass"], "axis2_reason": row["axis2_reason"],
        "overall_pass": row["overall_pass"],
    })


async def judge_one(session, sem, item, stats, write_lock):
    async with sem:
        qid = item["question_id"]
        messages = judge_prompt.build_judge_messages(item, item["question_text"])
        try:
            raw = await _call_llm(session, messages)
            verdict = parse_verdict(raw)
            if verdict is None:
                logger.warning("Unparseable/incomplete verdict for question_id=%d; skipping.", qid)
                stats["unparsed"] += 1
                return
            row = {
                "question_id": qid, "persona_id": item["persona_id"],
                "raw_response": raw, "model": LLM_CONFIG["model"], **verdict,
            }
            # Serialize DB+CSV writes through a single lock (no interleaving).
            async with write_lock:
                await asyncio.to_thread(_db_save, row, item, item["question_text"])
            stats["done"] += 1
            if verdict["overall_pass"]:
                stats["pass"] += 1
            else:
                stats["fail"] += 1
        except Exception:
            logger.exception("Judge failed for question_id=%d", qid)
            stats["errors"] += 1


def _fmt_eta(seconds) -> str:
    """Format seconds as H:MM:SS (or '—' if unknown)."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


async def _progress(stats, total, start, interval=10.0):
    """Emit a live progress line every `interval`s: done/total, %, rate, ETA."""
    while True:
        await asyncio.sleep(interval)
        handled = stats["done"] + stats["unparsed"] + stats["errors"]
        elapsed = max(asyncio.get_event_loop().time() - start, 1e-6)
        rate = handled / elapsed
        eta = (total - handled) / rate if rate > 0 and total else None
        pct = (100.0 * handled / total) if total else 100.0
        logger.info("PROGRESS %d/%d (%.1f%%) | pass=%d fail=%d unparsed=%d err=%d "
                    "| %.2f/s | ETA %s",
                    handled, total, pct, stats["pass"], stats["fail"],
                    stats["unparsed"], stats["errors"], rate, _fmt_eta(eta))


async def run(batch_size: int, limit: int | None):
    judge_db.init()
    db.init_pool()
    conn = db.get_connection()
    total_target = judge_db.count_unjudged(conn)
    if limit is not None:
        total_target = min(total_target, limit)
    logger.info("Unjudged questions to evaluate: %d (concurrency=%d, thinking=%s)",
                total_target, batch_size, JUDGE_CONFIG["thinking"])

    sem = asyncio.Semaphore(batch_size)
    write_lock = asyncio.Lock()
    stats = {"done": 0, "pass": 0, "fail": 0, "unparsed": 0, "errors": 0}
    remaining = limit

    start = asyncio.get_event_loop().time()
    progress_task = asyncio.create_task(_progress(stats, total_target, start))

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                fetch_n = batch_size if remaining is None else min(batch_size, remaining)
                if fetch_n <= 0:
                    break
                items = await asyncio.to_thread(judge_db.fetch_unjudged, conn, fetch_n)
                if not items:
                    break
                await asyncio.gather(*[
                    judge_one(session, sem, it, stats, write_lock) for it in items
                ])
                if remaining is not None:
                    remaining -= len(items)
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        db.put_connection(conn)
        db.close_pool()

    elapsed = asyncio.get_event_loop().time() - start
    logger.info("=== JUDGE COMPLETE in %s === done=%d pass=%d fail=%d unparsed=%d errors=%d",
                _fmt_eta(elapsed), stats["done"], stats["pass"], stats["fail"],
                stats["unparsed"], stats["errors"])


# ─── Report ──────────────────────────────────────────────────────────────────

def _fetchall(conn, sql):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return cur.fetchall()


def report():
    judge_db.init()
    db.init_pool()
    conn = db.get_connection()
    try:
        agg = _fetchall(conn, """
            SELECT COUNT(*) total,
                   COUNT(*) FILTER (WHERE overall_pass) overall_pass,
                   COUNT(*) FILTER (WHERE axis1_pass)   a1_pass,
                   COUNT(*) FILTER (WHERE axis2_pass)   a2_pass
            FROM judge_evaluations;
        """)[0]
        total = agg["total"] or 0
        print("\n" + "=" * 60)
        print("  JUDGE REPORT")
        print("=" * 60)
        print(f"\nEvaluated: {total}")
        if total:
            pct = lambda n: f"{n} ({100*n/total:.1f}%)"
            print(f"  overall PASS:          {pct(agg['overall_pass'])}")
            print(f"  Axis 1 (linguistic):   {pct(agg['a1_pass'])}")
            print(f"  Axis 2 (contextual):   {pct(agg['a2_pass'])}")

            print("\nWorst cohorts by overall FAIL rate (min 20 judged):")
            for dim in ("profession", "education", "pain_point"):
                print(f"\n  by {dim}:")
                rows = _fetchall(conn, f"""
                    SELECT p.json_metadata->>'{dim}' k,
                           COUNT(*) n,
                           ROUND(100.0*COUNT(*) FILTER (WHERE NOT e.overall_pass)/COUNT(*),1) fail_pct
                    FROM judge_evaluations e JOIN personas p USING (persona_id)
                    GROUP BY k HAVING COUNT(*) >= 20
                    ORDER BY fail_pct DESC LIMIT 6;
                """)
                for r in rows:
                    print(f"    {str(r['k'])[:34]:<34} fail={r['fail_pct']}%  (n={r['n']})")

            print("\nSample Axis-1 FAIL reasons:")
            for r in _fetchall(conn, """
                SELECT axis1_reason FROM judge_evaluations
                WHERE axis1_pass = FALSE AND axis1_reason <> '' LIMIT 5;
            """):
                print(f"    - {r['axis1_reason'][:110]}")
            print("\nSample Axis-2 FAIL reasons:")
            for r in _fetchall(conn, """
                SELECT axis2_reason FROM judge_evaluations
                WHERE axis2_pass = FALSE AND axis2_reason <> '' LIMIT 5;
            """):
                print(f"    - {r['axis2_reason'][:110]}")
        print("=" * 60 + "\n")
    finally:
        db.put_connection(conn)
        db.close_pool()


def main():
    ap = argparse.ArgumentParser(description="Two-axis LLM judge over generated questions.")
    ap.add_argument("--batch-size", type=int, default=JUDGE_CONFIG["concurrency"],
                    help="Concurrent judge requests (default from JUDGE_CONCURRENCY).")
    ap.add_argument("--limit", type=int, default=None, help="Judge at most N unjudged.")
    ap.add_argument("--report", action="store_true", help="Print pass-rate summary and exit.")
    args = ap.parse_args()

    if args.report:
        report()
        return
    asyncio.run(run(args.batch_size, args.limit))


if __name__ == "__main__":
    main()
