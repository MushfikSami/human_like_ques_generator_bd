# Workflow: Bangladeshi Persona & Human-Like Question Generation Agent (v2)

> Reworked pipeline. Supersedes the original linear workflow. Focuses on
> reliability (no infinite loops, no CSV races, pooled DB), diversity
> (coverage-driven personas + dedup), meaningful grading (separate judge call +
> deterministic gates), and quality reporting.

---

## Architecture at a glance

```
persona_generator ──► personas (status='pending')
                                 │
question_generator.run() ────────┤  fetch_pending (status='pending' only → terminates)
   per persona (async, semaphore):
     prompt_engine.build_prompt ─► LLM ─► cot_module.parse_draft
       └► cot_module.programmatic_checks (bengali_ratio + banned phrases)
       └► cot_module.build_cot_prompt ─► LLM judge ─► parse_judge (JSON verdict)
       └► on fail: build_rewrite_prompt ─► LLM (≤2 in-cycle rewrites)
       └► cot_module.dedup_hash (near-dup flag)
                                 │
             asyncio.Queue ─► single _writer task ─► db.save_question + append_to_csv
                                 │
   success → status='done' ;  exception → attempts++ ;  attempts≥MAX → status='failed'
```

Key invariants:
- **Termination:** `fetch_pending` returns only `status='pending'`. Every persona
  ends as `done` or `failed` (dead-letter after `GEN_MAX_ATTEMPTS`). No infinite loop.
- **No CSV races:** all DB + CSV writes go through ONE `_writer` task.
- **Bounded DB usage:** a `ThreadedConnectionPool` (not one conn per persona).

---

## Phase A — Config & schema

- `config.py` exposes `DB_CONFIG`, `LLM_CONFIG`, and **`GEN_CONFIG`**
  (`concurrency`, `pool_min/max`, `max_attempts`, `temperature`, `min_bengali_ratio`).
  All overridable via `.env` (`GEN_CONCURRENCY`, `GEN_MAX_ATTEMPTS`,
  `GEN_TEMPERATURE`, `GEN_MIN_BENGALI_RATIO`, `DB_POOL_MAX`, `LLM_URL`, …).
- Tables:
  - `personas` — adds `status VARCHAR ('pending'|'done'|'failed')`, `attempts INT`, `error TEXT`.
  - **`hlq_questions`** — our questions table. NOTE: `gov_spider_db` is shared with
    another project that owns a different `generated_questions` table, so we use a
    dedicated name to avoid collision. Columns include `dedup_hash`, `quality_flags JSONB`.
- `python main.py --init-db` creates tables + runs migrations. `--migrate` re-applies
  additive migrations (idempotent `ADD COLUMN IF NOT EXISTS`).

## Phase B — Coverage-driven personas (`persona_generator.py`)

- `_stratified_cells(count, rng)` walks `REGIONS × PROFESSIONS × PAIN_POINTS` with
  shuffled co-prime striding so all three dimensions are evenly covered (verified:
  200 personas touch 68 regions / 51 professions / 48 pain points).
- Remaining fields (age, gender, education, social status, backstory) are seeded
  per-persona → fully reproducible.
- `bulk_insert_personas` uses `execute_values` (page_size 500) instead of row-by-row.

## Phase C — Generation + meaningful grading

- `prompt_engine.build_prompt` — system prompt with the 7 human-like guidelines,
  outputs ONLY `<draft_questions>`. (Self-grading removed — grading is a separate call.)
- `cot_module`:
  - `parse_draft` — strips reasoning `<think>` blocks; discards pure-reasoning
    monologues so they can't be persisted as the "question".
  - `programmatic_checks` — deterministic gates: `bengali_ratio ≥ min_bengali_ratio`
    (Unicode block U+0980–U+09FF) + Bengali-aware banned-phrase list (formal/officialese
    Bengali + a few English tells; the old irrelevant English-word list is gone).
  - `build_cot_prompt` / `parse_judge` — a **separate judge completion** returning
    `{"verdict","is_bengali","reasons"}` JSON. Robust parse; lenient PASS only on
    unparseable judge (gates still apply).
  - `build_rewrite_prompt` — up to 2 in-cycle rewrites when a gate or the judge fails.
  - `dedup_hash` — NFKC-normalized SHA1 for near-dup detection.
- **Model note:** qwen3 reasons before answering; `LLM_MAX_TOKENS=3000` gives room to
  finish thinking AND emit the tags (1024 truncated mid-thought → empty output).

## Phase D — Reliable batch engine (`question_generator.py`)

- `run(batch_size)` → semaphore-limited async fan-out; DB work via `asyncio.to_thread`
  over the pool; single `_writer` consumes an `asyncio.Queue`.
- Dead-letter: on exception, `increment_attempt`; at `max_attempts`, `mark_failed`.
- Safe pause/resume: re-running `--generate` picks up remaining `pending` rows.

## Phase E — Quality metrics & dedup (`report.py`, `main.py --report`)

- Prints persona status counts, matrix coverage histograms (region/profession/pain),
  judge-fail %, programmatic-fail %, flagged-duplicate %, hash duplicate rate,
  avg length, avg Bengali ratio.

## Phase F — Reproducibility (honest)

- **Personas:** fully reproducible from the master seed (42/43).
- **Questions:** *approximately* reproducible only — vLLM continuous batching +
  `temperature>0` is not deterministic from `seed` alone. The **raw LLM response is
  stored in `cot_log`**, so any generation is exactly auditable even if not
  byte-reproducible. (This corrects the original "regenerate the exact question" claim.)

---

## Execution order

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `pip install -r requirements.txt` | Deps |
| 2 | edit `.env` | Credentials + tuning |
| 3 | `python main.py --init-db` | Create tables + migrate |
| 4 | `python main.py --gen-personas 200` | Seed test personas (stratified) |
| 5 | `python main.py --generate --batch-size 10` | Generate (micro-batch) |
| 6 | `python main.py --report` | Coverage + quality audit |
| 7 | tune `.env` (`GEN_TEMPERATURE`, `GEN_MIN_BENGALI_RATIO`) | Iterate |
| 8 | `python main.py --gen-personas 25000` | Full persona set |
| 9 | `python main.py --generate` | Full run (resumable) |
| 10 | `python main.py --report` | Final metrics |

## Verified behaviours (this rework)

- Even matrix coverage; reproducible personas.
- Live run produces authentic Bengali (`bengali_ratio ≈ 0.96`, judge PASS).
- Dead-letter: bad LLM URL → run **terminates**, persona → `status='failed'` (no hang).
- Report populates all metrics.
