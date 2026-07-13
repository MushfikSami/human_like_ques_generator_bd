# Plan: Expand persona_generator.py coverage (weighted regions + education striding)

## Context (this task)
The reworked pipeline is done and working. The user now wants richer demographic
coverage in [persona_generator.py](persona_generator.py): modern gig/micro-professions,
NRB/migrant demographics, hyper-local/marginalized environments (slums, camps,
char areas), and modern digital pain points (Universal Pension, e-TIN, BDRIS, etc.).

Architecturally, regions become a **weighted probabilistic** dimension (so Dhaka
Metro keeps a realistic ~25% share while slums/camps get small 1-3% shares), which
means region can no longer sit in the deterministic co-prime striding matrix. It is
swapped out and **education** is swapped in, so the matrix strides over
`professions × pain_points × education_levels`.

Verified: with the projected array sizes (professions 62, pains 55, education 9),
the multipliers **7, 13, 11 are each coprime** with their array length, so striding
still yields even per-axis coverage. Trade-off (intended): forcing education via the
matrix replaces its realistic `EDUCATION_WEIGHTS` distribution with ~uniform
coverage; the weights are retained only for the legacy `forced_cell=None` path.

## Changes — all in persona_generator.py
1. **Data expansion**
   - `PROFESSIONS` += gig/micro (Foodpanda/Pathao rider, F-commerce/Facebook-Live
     seller, bKash/Nagad agent, Upwork/Fiverr graphic designer, YouTube/TikTok
     creator, Hijra community member) and NRB/migrant (Dubai construction worker,
     Malaysia palm-oil worker, Saudi convenience-store clerk, student in N.America/UK,
     returnee migrant worker) — 11 new.
   - `PAIN_POINTS` += Universal Pension (Prottoy/Surokkha) enrollment, e-TIN
     registration & zero-return, BDRIS server downtime, BRTA smart-license biometric
     delay, Probashi Kallyan Bank loan, bKash-fraud/cybercrime report to DB police,
     dual-citizenship certificate for e-Passport — 7 new.
2. **Weighted regions** — add hyper-local entries (Korail Slum, Bhashantek Bosti,
   Geneva Camp (Mohammadpur), Railway Colony (Chattogram), Sitakunda Shipbreaking
   Yard Area, Char areas of Kurigram) to `REGIONS`, and add a parallel
   `REGION_WEIGHTS` list (same length/order). Weighting scheme: Dhaka ~25, other
   divisions/major cities higher, ordinary districts ~1 each, hyper-local slums/camps
   small (~1-3). `rng.choices` normalizes automatically, so weights are relative.
   Add an assert `len(REGIONS) == len(REGION_WEIGHTS)` guard.
3. **`_stratified_cells`** — drop region; shuffle+stride `professions` (`i*7`),
   `pains` (`i*13`), `education` (`i*11`); return `(profession, pain_point, education)`
   tuples. Update docstring.
4. **`_generate_single_persona`** — `forced_cell` now unpacks
   `(profession, pain_point, education)`; sample `location` probabilistically via
   `rng.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0]` in BOTH paths; keep the
   weighted `education` sampling only when `forced_cell is None`. Update docstring.
5. Module docstring: note region is weighted-probabilistic, education is strided.
   All selections keep using the seeded `rng` → reproducibility preserved.

## Downstream
No schema or callsite changes: `generate_personas` still passes `cell` to
`_generate_single_persona`; `report.py` histograms (`location/profession/pain_point`)
still work. Optional nicety (not required): add an `education` histogram to
[report.py](report.py) `print_report`.

## Verification
1. `python -c "import math,persona_generator as p; assert len(p.REGIONS)==len(p.REGION_WEIGHTS); print(len(p.PROFESSIONS),len(p.PAIN_POINTS),[math.gcd(m,n) for m,n in [(7,len(p.PROFESSIONS)),(13,len(p.PAIN_POINTS)),(11,len(p.EDUCATION_LEVELS))]])"` → gcds all 1.
2. Reproducibility: `_generate_single_persona(123, forced_cell=('tailor','e-TIN registration','SSC pass'))` equals itself on a second call; education field == forced value.
3. Coverage on a dry run of `_stratified_cells(2000, random.Random(43))`: every
   profession, pain_point, and education level appears; region distribution from a
   sample of 2000 personas shows Dhaka dominant and each hyper-local area present.
4. Optional live: regenerate a small persona batch into a scratch/empty table and
   `python main.py --report` to eyeball histograms. (Do NOT re-run `--gen-personas`
   against a table that already holds the 25k — it appends duplicates.)

---

# Plan: Rework the Human-Like Question Generator (workflow + code)

## Context
The project (`workplan.md` → `workflow.md` → implemented Python) generates 25,000
Bangladeshi personas and, for each, produces human-like Bengali questions for a
government-service chatbot via a local vLLM (`qwen3-35b-awq`) endpoint, storing
results in PostgreSQL + CSV.

The user asked whether a better workflow is possible. After reading the actual
implementation (not just the docs), the answer is yes: there are **real
correctness bugs** and several **design weaknesses** that undermine the stated
goals (reliable full run, diversity, reproducibility, quality). The user chose a
**full rethink**: fix the code AND rewrite `workflow.md`, adding diversity/dedup,
coverage-driven persona generation, structured outputs, and quality metrics.

This plan defines the improved workflow and the concrete code changes.

## Real bugs to fix (verified in code)
1. **Infinite loop on persistent failure** — [question_generator.py:200-202](question_generator.py#L200-L202)
   failed personas are never marked, so [question_generator.py:233-235](question_generator.py#L233-L235)
   re-fetches the same failing batch forever. A single un-generatable persona
   stalls the whole 25k run.
2. **Concurrent CSV corruption** — `append_to_csv` ([db.py:200-210](db.py#L200-L210))
   is called from up to 50 concurrent tasks with no lock → interleaved/garbled rows.
3. **Connection-per-persona, no pooling** — [question_generator.py:248-251](question_generator.py#L248-L251)
   opens a fresh psycopg2 connection per persona; exhausts Postgres at scale
   despite the README claiming pooling.
4. **Dead CoT code / weak self-grading** — `build_cot_prompt` ([cot_module.py:96](cot_module.py#L96))
   is never called. The inline reflection in `SYSTEM_PROMPT` runs but a model
   grading its own output almost always self-reports `Verdict: PASS`, so the
   rewrite path rarely fires.
5. **Brittle/irrelevant banned-word gate** — [cot_module.py:23-29](cot_module.py#L23-L29)
   bans English words (`regarding`, `therefore`, `hence`) that essentially never
   appear in Bengali output, so the gate is near-inert.
6. **Overpromised reproducibility** — `temperature=0.8` under vLLM continuous
   batching is not deterministic from `seed` alone; the "regenerate the exact
   question from persona_id+seed" guarantee (workplan Phase 8/11) won't hold.
7. **No diversity control** — random sampling ([persona_generator.py:181-191](persona_generator.py#L181-L191))
   gives no coverage guarantee across the region×profession×pain-point matrix,
   and there is no dedup of generated questions → mode collapse at 25k.

## Improved workflow (new `workflow.md`)
Restructure into these phases, replacing the current linear one:

- **Phase A — Config & schema.** Add new columns to `personas`:
  `status VARCHAR DEFAULT 'pending'` (pending/done/failed), `attempts INT DEFAULT 0`,
  `error TEXT`. Add to `generated_questions`: `dedup_hash TEXT`, `quality_flags JSONB`.
  Keep `.env`/`config.py` as-is; add `MAX_ATTEMPTS`, `CONCURRENCY`, `POOL_SIZE`,
  `TEMPERATURE` config keys.
- **Phase B — Coverage-driven personas.** Replace pure random with stratified
  sampling over the `REGIONS × PROFESSIONS × PAIN_POINTS` matrix so the 25k
  spread is even (round-robin / quota per cell), still seeded for reproducibility.
  Bulk insert with `execute_values` instead of row-by-row.
- **Phase C — Generation with a two-stage prompt.** Keep a single generator call
  but make grading meaningful: (1) generate draft, (2) a **separate judge call**
  (repurpose the existing `build_cot_prompt`) that returns structured JSON
  `{verdict, reasons, is_bengali}`; rewrite only when judge fails or programmatic
  checks fail. Programmatic checks = Bengali-script ratio (regex over Unicode
  Bengali block) + a *trimmed, Bengali-aware* banned-phrase list.
- **Phase D — Reliable batch engine.** Dead-letter handling (mark `failed` after
  `MAX_ATTEMPTS`, loop fetches only `status='pending'` and terminates), a psycopg2
  `ThreadedConnectionPool`, and a **single writer task** consuming an `asyncio.Queue`
  for both DB and CSV (no races).
- **Phase E — Quality metrics & dedup.** Compute a normalized `dedup_hash` per
  question; skip/flag near-duplicates. Emit a run report: coverage histogram,
  duplicate rate, judge-fail rate, avg length, % non-Bengali. `main.py --report`.
- **Phase F — Honest reproducibility.** Document that persona generation is fully
  reproducible; question generation is *approximately* reproducible (record the
  raw response so audits are exact). Update workplan/README claims accordingly.

## Code changes (files)
- **config.py** — add `GEN_CONFIG` (concurrency, pool size, max attempts, temp).
- **db.py** — add `ThreadedConnectionPool` (`get_pool`, `put_conn`); migrate schema
  (new columns via `ALTER TABLE ... IF NOT EXISTS` pattern); change
  `fetch_unprocessed_personas` → `fetch_pending`; add `mark_failed`,
  `increment_attempt`; make `append_to_csv` callable only from the single writer.
- **persona_generator.py** — stratified `generate_personas`; batch insert.
- **prompt_engine.py** — keep system prompt; drop the redundant inline reflection
  (grading moves to the judge call); tighten wording.
- **cot_module.py** — wire up `build_cot_prompt` as the real judge; parse
  structured verdict; replace banned-word list with a Bengali-aware check +
  `bengali_ratio()` helper; add `dedup_hash()`.
- **question_generator.py** — pool-based conns; retry/attempt loop with dead-letter;
  `asyncio.Queue` + single `writer_task` for DB+CSV; terminate when no pending rows.
- **main.py** — add `--report`; keep existing flags; `--migrate` to apply new columns.
- **workflow.md / workplan.md / README.md** — rewrite per phases above; fix the
  reproducibility claims.

## Verification
1. `python main.py --init-db --migrate` — confirm new columns exist (`\d personas`).
2. `python main.py --gen-personas 200` — check coverage histogram is even across
   regions/professions (via `--report`).
3. Point `LLM_URL` at the running vLLM; `python main.py --generate --batch-size 10`
   on the 200 test personas. Confirm: run **terminates** (no infinite loop) even if
   some personas fail; CSV rows are well-formed (no interleaving); failed personas
   show `status='failed'` with `attempts=MAX_ATTEMPTS`.
4. Inject a forced failure (bad LLM URL for a subset) to prove dead-letter path
   ends the loop.
5. `python main.py --report` — verify duplicate rate, judge-fail rate, non-Bengali
   %, coverage all populate.
6. Re-run one persona twice by seed — persona identical; question response stored
   raw for audit even if not byte-identical.
7. `python -c "import ast; [ast.parse(open(f).read()) for f in (...)]"` sanity, plus
   spot-run each CLI flag.
