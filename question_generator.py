"""
question_generator.py — Core Generation Loop

Processes 'pending' personas from the database, generates Bengali questions via
the vLLM endpoint, grades them with a separate judge call plus deterministic
quality gates, and persists results.

Reliability design (fixes over the previous version):
  * Dead-letter: a persona is retried up to GEN_CONFIG["max_attempts"], then
    marked status='failed'. The fetch loop only returns status='pending', so a
    persistently-failing persona can never stall the full run — it terminates.
  * Connection pool: DB work borrows from a shared ThreadedConnectionPool
    (via asyncio.to_thread) instead of opening one connection per persona.
  * Single writer: all DB writes and CSV appends flow through ONE writer task
    fed by an asyncio.Queue, eliminating the concurrent-CSV corruption race.
"""

import asyncio
import json
import logging
import time
from datetime import datetime

import aiohttp

import db
import prompt_engine
import cot_module
import memory_store
from config import LLM_CONFIG, GEN_CONFIG, MEMORY_CONFIG

logger = logging.getLogger(__name__)

# Thinking is disabled on every call (see _call_llm), so the model emits the
# answer directly and needs far less headroom. Bengali questions / judge JSON are
# short; 800 tokens is ample.
LLM_MAX_TOKENS = 800

# Sentinel to stop the writer task.
_STOP = object()


async def _call_llm(session, messages, seed=None):
    """Send a chat completion request to the vLLM endpoint with retry/backoff."""
    payload = {
        "model": LLM_CONFIG["model"],
        "messages": messages,
        "temperature": GEN_CONFIG["temperature"],
        "max_tokens": LLM_MAX_TOKENS,
        # Disable qwen3's chain-of-thought so it answers directly. This is the
        # vLLM-honored switch (the in-prompt "/no_think" text is not reliable)
        # and cuts generated tokens ~5-10x.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(3):
        try:
            async with session.post(
                LLM_CONFIG["url"], json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("LLM API error (status %d, attempt %d): %s",
                                 response.status, attempt + 1, error_text)
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(f"LLM API error: {response.status}")
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.warning("LLM call failed (attempt %d/3): %s", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise


def _metadata(persona: dict) -> dict:
    md = persona.get("json_metadata")
    if isinstance(md, str):
        return json.loads(md)
    if isinstance(md, dict):
        return md
    return {}


# ─── Per-persona pipeline ────────────────────────────────────────────────────

async def generate_for_persona(session, semaphore, persona, out_queue, seen_hashes,
                                memory=None):
    """
    Generate, grade, and (on success) enqueue one persona's question.

    Returns one of: "done", "retry", "failed" — the run loop uses this to
    decide DB status transitions.

    `memory` (optional MemoryStore) supplies procedural-memory context to the
    prompt and a semantic near-duplicate signal that feeds the rewrite loop.
    """
    persona_id = persona["persona_id"]
    meta = _metadata(persona)
    random_seed = meta.get("random_seed", persona_id)

    # Procedural-memory context (avoid-openers + exemplars) — computed once.
    mem_ctx = memory.get_context(meta) if memory else None

    async with semaphore:
        try:
            # 1. Generate draft
            messages = prompt_engine.build_prompt(persona, memory_context=mem_ctx)
            logger.info("Generating persona_id=%d (%s / %s)", persona_id,
                        persona.get("profession", "?"), persona.get("location", "?"))
            response = await _call_llm(session, messages, seed=random_seed)
            draft = cot_module.parse_draft(response)

            cot_parts = [f"=== RAW ===\n{response}"]

            # 2. Gate + (conditional) judge, with in-cycle rewrites.
            #    - The judge LLM call only runs when the cheap deterministic gates
            #      already pass; a draft that fails the gate is rewritten without
            #      wasting a judge call on it.
            #    - We KEEP THE BEST attempt (never save a worse rewrite), ranked
            #      by (passed, bengali_ratio).
            #    - Embedding / near-dup happen ONCE on the final text (below), not
            #      per iteration — far fewer embeds.
            max_rewrites = GEN_CONFIG["max_rewrites"]
            min_ratio = GEN_CONFIG["min_bengali_ratio"]
            candidate = draft
            best = None  # {"text","flags","key"}
            for rw in range(max_rewrites + 1):
                ok, flags = cot_module.programmatic_checks(candidate, min_ratio)

                if ok:
                    judge_raw = await _call_llm(
                        session, cot_module.build_cot_prompt(candidate, persona),
                        seed=random_seed + 1000 + rw)
                    judge = cot_module.parse_judge(judge_raw)
                    cot_parts.append(
                        f"=== GATE {rw} ===\n{flags}\n=== JUDGE {rw} (raw) ===\n"
                        f"{judge_raw}\n=== JUDGE {rw} (parsed) ===\n{judge}")
                else:
                    # Gate failed — skip the judge, we already know it needs work.
                    judge = {"verdict": "SKIPPED", "is_bengali": False,
                             "reasons": [f"gate fail: {flags}"]}
                    cot_parts.append(f"=== GATE {rw} (FAIL, judge skipped) ===\n{flags}")

                passed = ok and judge["verdict"] == "PASS" and judge["is_bengali"]
                cand_flags = {**flags, "judge": judge}

                key = (1 if passed else 0, float(flags.get("bengali_ratio", 0.0)))
                if best is None or key > best["key"]:
                    best = {"text": candidate, "flags": cand_flags, "key": key}

                if passed or rw == max_rewrites:
                    break

                # Rewrite — escalate hard when the failure is romanized output.
                lang_fail = float(flags.get("bengali_ratio", 0.0)) < min_ratio
                problems = "; ".join(judge["reasons"]) or str(flags)
                rewrite_msgs = cot_module.build_rewrite_prompt(
                    candidate, problems, persona, lang_fail=lang_fail)
                rw_resp = await _call_llm(session, rewrite_msgs, seed=random_seed + 2000 + rw)
                candidate = cot_module.parse_draft(rw_resp)
                cot_parts.append(f"=== REWRITE {rw} (lang_fail={lang_fail}) ===\n{rw_resp}")

            # Keep the best candidate across all attempts.
            final_text = best["text"]
            quality_flags = best["flags"]
            cot_parts.append(f"=== SELECTED best key={best['key']} ===")

            # 3. Embed ONCE (final text) — for storage + a single near-dup flag.
            vec = None
            near_dup = False
            if memory:
                vec = await asyncio.to_thread(memory.embed, final_text)
                near_dup = memory.is_near_dup(vec, meta)
            quality_flags["near_dup"] = near_dup

            # Dedup flag (exact hash) + memory bookkeeping fields
            dh = cot_module.dedup_hash(final_text)
            is_dup = dh in seen_hashes
            seen_hashes.add(dh)
            quality_flags["duplicate"] = is_dup

            emb_bytes = vec.tobytes() if vec is not None else None
            opener = memory.opener(final_text) if memory else None
            quality_score = (memory_store.score_from_flags(quality_flags)
                             if memory else None)

            # 4. Enqueue for the single writer
            await out_queue.put({
                "persona": persona,
                "persona_id": persona_id,
                "question_text": final_text,
                "cot_log": "\n\n".join(cot_parts),
                "random_seed": random_seed,
                "dedup_hash": dh,
                "quality_flags": quality_flags,
                "embedding": emb_bytes,
                "opener": opener,
                "quality_score": quality_score,
                "vec": vec,          # np array, for in-RAM memory.record
                "meta": meta,
            })
            return "done"

        except Exception as e:
            logger.exception("Generation failed for persona_id=%d", persona_id)
            # Decide retry vs dead-letter based on prior attempts.
            attempts = (persona.get("attempts") or 0) + 1
            await asyncio.to_thread(_db_increment_attempt, persona_id)
            if attempts >= GEN_CONFIG["max_attempts"]:
                await asyncio.to_thread(_db_mark_failed, persona_id, str(e))
                return "failed"
            return "retry"


# ─── DB thread helpers (borrow/return pooled connections) ────────────────────

def _db_increment_attempt(persona_id):
    conn = db.get_connection()
    try:
        db.increment_attempt(conn, persona_id)
    finally:
        db.put_connection(conn)


def _db_mark_failed(persona_id, error):
    conn = db.get_connection()
    try:
        db.mark_failed(conn, persona_id, error)
    finally:
        db.put_connection(conn)


def _db_save(item):
    conn = db.get_connection()
    try:
        db.save_question(conn, item["persona_id"], item["question_text"],
                         item["cot_log"], item["random_seed"],
                         dedup_hash=item["dedup_hash"],
                         quality_flags=item["quality_flags"],
                         embedding=item.get("embedding"),
                         opener=item.get("opener"),
                         quality_score=item.get("quality_score"))
    finally:
        db.put_connection(conn)


# ─── Single writer task ──────────────────────────────────────────────────────

async def _writer(queue: asyncio.Queue, stats: dict, memory=None):
    """Consume results and persist them to DB + CSV, serially (no races)."""
    while True:
        item = await queue.get()
        if item is _STOP:
            queue.task_done()
            break
        try:
            await asyncio.to_thread(_db_save, item)
            persona = item["persona"]
            db.append_to_csv({
                "persona_id": item["persona_id"],
                "age": persona.get("age"),
                "gender": persona.get("gender"),
                "location": persona.get("location"),
                "profession": persona.get("profession"),
                "social_status": persona.get("social_status"),
                "backstory": persona.get("backstory", ""),
                "question_text": item["question_text"],
                "cot_log": item["cot_log"][:500],
                "random_seed": item["random_seed"],
                "created_at": datetime.now().isoformat(),
            })
            # Update in-RAM procedural memory (serial → safe).
            if memory and item.get("meta") is not None:
                memory.record(item["meta"], item["question_text"], item.get("vec"),
                              item.get("quality_score") or 0.0, opener=item.get("opener"))
            stats["success"] += 1
            if item["quality_flags"].get("duplicate"):
                stats["duplicates"] += 1
            if item["quality_flags"].get("near_dup"):
                stats["near_dups"] = stats.get("near_dups", 0) + 1
        except Exception:
            logger.exception("Writer failed to persist persona_id=%d", item["persona_id"])
            stats["write_errors"] += 1
        finally:
            queue.task_done()


# ─── Progress reporter ───────────────────────────────────────────────────────

def _fmt_eta(seconds: float) -> str:
    """Format seconds as H:MM:SS (or '—' if unknown)."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


async def _progress(stats: dict, total: int, start: float,
                    interval: float = 10.0):
    """
    Periodically emit a single progress line: count/total, %, rate, ETA.

    Runs until cancelled. `total` is the number of personas pending at start.
    """
    while True:
        await asyncio.sleep(interval)
        done = stats["success"] + stats["failed"]
        elapsed = max(time.monotonic() - start, 1e-6)
        rate = done / elapsed
        remaining = max(total - done, 0)
        eta = remaining / rate if rate > 0 else None
        pct = (100.0 * done / total) if total else 100.0
        logger.info(
            "PROGRESS %d/%d (%.1f%%) | ok=%d fail=%d dup=%d | %.2f/s | ETA %s",
            done, total, pct, stats["success"], stats["failed"],
            stats["duplicates"], rate, _fmt_eta(eta),
        )


# ─── Main loop ───────────────────────────────────────────────────────────────

async def run(batch_size: int = None):
    """Process all 'pending' personas until none remain, then terminate."""
    batch_size = batch_size or GEN_CONFIG["concurrency"]
    logger.info("Starting question generation (batch_size=%d)...", batch_size)

    db.init_pool()
    semaphore = asyncio.Semaphore(batch_size)
    queue: asyncio.Queue = asyncio.Queue(maxsize=batch_size * 2)
    seen_hashes: set = set()
    stats = {"success": 0, "duplicates": 0, "near_dups": 0, "write_errors": 0,
             "retry": 0, "failed": 0}

    fetch_conn = db.get_connection()

    # Crash recovery: release any personas stranded in 'processing' by a
    # previous interrupted run back to 'pending'.
    stranded = await asyncio.to_thread(db.reset_processing, fetch_conn)
    if stranded:
        logger.info("Reset %d stranded 'processing' personas to pending.", stranded)

    # Procedural memory: build + prime from prior questions (best-effort).
    memory = None
    if MEMORY_CONFIG.get("enabled"):
        memory = memory_store.MemoryStore(MEMORY_CONFIG)
        try:
            await asyncio.to_thread(memory.embed, "warmup")   # load model up front
            await asyncio.to_thread(memory.prime, fetch_conn)
        except Exception:
            logger.exception("Memory prime failed; continuing without memory.")
            memory = None

    writer_task = asyncio.create_task(_writer(queue, stats, memory))

    # Total work to do = personas pending at the start of this run.
    total = await asyncio.to_thread(db.count_personas, fetch_conn, "pending")
    logger.info("Personas to process this run: %d", total)
    start = time.monotonic()
    progress_task = asyncio.create_task(_progress(stats, total, start))

    batch_no = 0
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                personas = await asyncio.to_thread(db.fetch_pending, fetch_conn, batch_size)
                if not personas:
                    logger.info("No more pending personas. Complete.")
                    break

                tasks = [generate_for_persona(session, semaphore, p, queue,
                                              seen_hashes, memory)
                         for p in personas]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # No barrier needed: fetch_pending atomically claims rows
                # (pending → processing), so the next fetch can never re-grab a
                # persona that's still in flight. Batches pipeline freely.
                for r in results:
                    if r == "retry":
                        stats["retry"] += 1
                    elif r == "failed":
                        stats["failed"] += 1

                batch_no += 1
                done = stats["success"] + stats["failed"]
                logger.info("Batch %d done | %d/%d | ok=%d fail=%d dup=%d",
                            batch_no, done, total, stats["success"],
                            stats["failed"], stats["duplicates"])
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        await queue.put(_STOP)
        await writer_task
        db.put_connection(fetch_conn)
        db.close_pool()

    elapsed = time.monotonic() - start
    logger.info("=== COMPLETE in %s === ok=%d dup=%d near_dup=%d failed=%d write_errors=%d",
                _fmt_eta(elapsed), stats["success"], stats["duplicates"],
                stats["near_dups"], stats["failed"], stats["write_errors"])
    return stats
