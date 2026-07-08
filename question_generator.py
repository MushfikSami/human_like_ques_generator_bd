"""
question_generator.py — Core Generation Loop

Processes batches of unprocessed personas from the database, sends them to the
LLM endpoint for question generation with Chain-of-Thought self-reflection,
and stores results in both PostgreSQL and CSV.

Uses asyncio with a semaphore to control concurrency (default 50 concurrent
requests) to avoid overwhelming the vLLM server.

The `processed` flag in the personas table ensures safe pause/resume —
the script can be stopped and restarted without duplicating work.
"""

import asyncio
import json
import logging
from datetime import datetime

import aiohttp

import db
import prompt_engine
import cot_module
from config import LLM_CONFIG

logger = logging.getLogger(__name__)

# Maximum number of rewrite attempts if the CoT module flags AI-sounding text
MAX_REWRITE_ATTEMPTS = 2

# LLM generation parameters
LLM_TEMPERATURE = 0.8
LLM_MAX_TOKENS = 1024


async def _call_llm(session: aiohttp.ClientSession, messages: list[dict],
                     seed: int | None = None) -> str:
    """
    Send a chat completion request to the vLLM endpoint.

    Args:
        session: Active aiohttp session.
        messages: OpenAI-compatible messages list.
        seed: Optional random seed for reproducibility.

    Returns:
        The generated text content from the LLM.

    Raises:
        Exception: If the API call fails after all retries.
    """
    payload = {
        "model": LLM_CONFIG["model"],
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if seed is not None:
        payload["seed"] = seed

    # Retry logic for transient failures
    for attempt in range(3):
        try:
            async with session.post(
                LLM_CONFIG["url"],
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("LLM API error (status %d, attempt %d): %s",
                                 response.status, attempt + 1, error_text)
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise Exception(f"LLM API error: {response.status} - {error_text}")

                result = await response.json()
                return result["choices"][0]["message"]["content"]

        except asyncio.TimeoutError:
            logger.warning("LLM call timed out (attempt %d/3)", attempt + 1)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except aiohttp.ClientError as e:
            logger.warning("LLM connection error (attempt %d/3): %s", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise


async def generate_for_persona(session: aiohttp.ClientSession,
                                semaphore: asyncio.Semaphore,
                                persona: dict,
                                db_conn) -> bool:
    """
    Full generation pipeline for a single persona.

    Steps:
    1. Build the prompt with persona details.
    2. Call the LLM with the persona's random_seed.
    3. Validate via CoT self-reflection.
    4. If flagged, attempt rewrites (up to MAX_REWRITE_ATTEMPTS).
    5. Save the final question and CoT log to DB + CSV.

    Args:
        session: Active aiohttp session for LLM calls.
        semaphore: Concurrency limiter.
        persona: Dict with persona details from the database.
        db_conn: Active psycopg2 connection.

    Returns:
        True if generation succeeded, False otherwise.
    """
    persona_id = persona["persona_id"]

    # Extract random_seed from json_metadata
    if isinstance(persona.get("json_metadata"), str):
        metadata = json.loads(persona["json_metadata"])
    elif isinstance(persona.get("json_metadata"), dict):
        metadata = persona["json_metadata"]
    else:
        metadata = {}

    random_seed = metadata.get("random_seed", persona_id)

    async with semaphore:
        try:
            # Step 1: Build the initial prompt
            messages = prompt_engine.build_prompt(persona)

            # Step 2: Call the LLM
            logger.info("Generating questions for persona_id=%d (%s from %s)",
                        persona_id, persona.get("profession", "?"),
                        persona.get("location", "?"))

            llm_response = await _call_llm(session, messages, seed=random_seed)

            # Step 3: Validate and refine via CoT
            question_text, cot_log = cot_module.validate_and_refine(llm_response, persona)

            # Step 4: Handle rewrites if needed
            rewrite_count = 0
            full_cot_log = cot_log

            while cot_log.startswith("NEEDS_REWRITE|") and rewrite_count < MAX_REWRITE_ATTEMPTS:
                rewrite_count += 1
                logger.info("Rewriting for persona_id=%d (attempt %d/%d)",
                            persona_id, rewrite_count, MAX_REWRITE_ATTEMPTS)

                # Extract the actual cot_log (after the NEEDS_REWRITE| prefix)
                actual_cot = cot_log.split("|", 1)[1]

                # Build the rewrite prompt
                rewrite_messages = cot_module.build_rewrite_prompt(
                    question_text,
                    f"Rewrite attempt {rewrite_count}",
                    persona,
                )

                # Call the LLM again for rewrite
                rewrite_response = await _call_llm(session, rewrite_messages, seed=random_seed + rewrite_count)
                question_text, cot_log = cot_module.validate_and_refine(rewrite_response, persona)

                full_cot_log += f"\n\n=== REWRITE ATTEMPT {rewrite_count} ===\n{cot_log}"

            # Clean the final cot_log (remove NEEDS_REWRITE prefix if still present)
            if full_cot_log.startswith("NEEDS_REWRITE|"):
                full_cot_log = full_cot_log.split("|", 1)[1]
                logger.warning("persona_id=%d still flagged after %d rewrites, saving anyway.",
                               persona_id, MAX_REWRITE_ATTEMPTS)

            # Step 5: Save to DB and CSV
            db.save_question(db_conn, persona_id, question_text, full_cot_log, random_seed)

            # Build CSV row with combined persona + question data
            csv_row = {
                "persona_id": persona_id,
                "age": persona.get("age"),
                "gender": persona.get("gender"),
                "location": persona.get("location"),
                "profession": persona.get("profession"),
                "social_status": persona.get("social_status"),
                "backstory": persona.get("backstory", ""),
                "question_text": question_text,
                "cot_log": full_cot_log[:500],  # Truncate for CSV readability
                "random_seed": random_seed,
                "created_at": datetime.now().isoformat(),
            }
            db.append_to_csv(csv_row)

            logger.info("✓ persona_id=%d — question saved (rewrites: %d)",
                        persona_id, rewrite_count)
            return True

        except Exception:
            logger.exception("✗ Failed to generate for persona_id=%d", persona_id)
            return False


async def run(batch_size: int = 50):
    """
    Main generation loop — processes all unprocessed personas in batches.

    Fetches batches of unprocessed personas from the database, generates
    questions for each using async concurrency (limited by semaphore),
    and continues until no unprocessed personas remain.

    The `processed` flag ensures safe pause/resume across script restarts.

    Args:
        batch_size: Number of concurrent personas to process per batch.
                    Also used as the semaphore limit.
    """
    logger.info("Starting question generation (batch_size=%d)...", batch_size)

    # Use a semaphore to limit concurrent LLM calls
    semaphore = asyncio.Semaphore(batch_size)

    # Create a shared DB connection for the main loop
    conn = db.get_connection()

    total_processed = 0
    total_success = 0
    total_failed = 0

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                # Fetch next batch of unprocessed personas
                personas = db.fetch_unprocessed_personas(conn, batch_size)

                if not personas:
                    logger.info("No more unprocessed personas. Generation complete.")
                    break

                logger.info("Processing batch of %d personas (total so far: %d)",
                            len(personas), total_processed)

                # Create tasks for each persona in the batch
                # Each persona gets its own DB connection for transactional safety
                tasks = []
                persona_conns = []
                for persona in personas:
                    p_conn = db.get_connection()
                    persona_conns.append(p_conn)
                    task = generate_for_persona(session, semaphore, persona, p_conn)
                    tasks.append(task)

                # Run all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Close per-persona connections
                for p_conn in persona_conns:
                    try:
                        p_conn.close()
                    except Exception:
                        pass

                # Tally results
                for result in results:
                    total_processed += 1
                    if isinstance(result, Exception):
                        total_failed += 1
                        logger.error("Task exception: %s", result)
                    elif result:
                        total_success += 1
                    else:
                        total_failed += 1

                logger.info("Batch complete. Success: %d, Failed: %d, Total: %d",
                            total_success, total_failed, total_processed)

    finally:
        conn.close()

    logger.info("=== GENERATION COMPLETE ===")
    logger.info("Total processed: %d | Success: %d | Failed: %d",
                total_processed, total_success, total_failed)
