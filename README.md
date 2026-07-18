# Bangladeshi Persona & Human-Like Question Generation Agent

A pipeline for generating 25,000 diverse Bangladeshi personas and producing
human-like Bengali questions about government services using a local LLM
(Qwen3-35B via vLLM). Quality is enforced by deterministic gates (Bengali-script
ratio + banned-phrase list) plus a **separate LLM judge call**, and diversity is
maintained by a GPU-backed **procedural-memory** layer (anti-repetition +
few-shot exemplars). Runs on a reliable, resumable batch engine.
See `workflow.md` and `docs/procedural_memory_design.md` for the full architecture.

### Key capabilities
- **Coverage-driven personas** — stratified striding over
  profession × pain-point × education (even coverage), weighted-probabilistic
  regions (realistic distribution incl. slums/camps/char areas), fully
  seed-reproducible.
- **Human-like generation** — 7 human-likeness guidelines, Bengali-script output,
  qwen3 "thinking" disabled for speed.
- **Quality control** — deterministic gates + a separate LLM judge; keep-the-best
  across up to 3 rewrites; forceful anti-romanization escalation.
- **Procedural memory** — a local `sentence-transformers` embedder (on GPU)
  powers overused-opener avoidance, few-shot style exemplars, and semantic
  near-duplicate rejection.
- **Reliable engine** — connection pool, single-writer queue (no CSV races),
  dead-letter retries, per-batch barrier (exactly one question per persona),
  safe pause/resume, live progress + ETA.
- **Reporting** — `--report` (coverage + quality + diversity) and
  `--topic-report` (self-contained HTML topic-distribution visual).

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env   # Edit with your DB password and LLM endpoint

# 3. Create database tables
python main.py --init-db

# 4. Generate test personas (micro-batch)
python main.py --gen-personas 100

# 5. Run micro-batch QA
python main.py --test

# 6. Review output
cat personas_questions.csv
```

## Project Structure

```
├── config.py              # DB + LLM + GEN + MEMORY settings (from .env)
├── db.py                  # PostgreSQL pool, migrations, transactional writes, CSV backup
├── persona_generator.py   # Stratified persona creation (74 regions, 62 professions, 55 topics)
├── prompt_engine.py       # System prompt (7 guidelines) + memory-context injection
├── cot_module.py          # Quality gates, LLM judge, rewrite loop, dedup
├── memory_store.py        # Procedural memory: GPU embedder, openers, exemplars, near-dup
├── question_generator.py  # Async engine: pool, dead-letter, single-writer queue, progress
├── report.py              # Coverage + quality + diversity report (--report)
├── topic_report.py        # Self-contained HTML topic-distribution report (--topic-report)
├── main.py                # CLI entry-point
├── requirements.txt       # Python dependencies (incl. sentence-transformers)
├── docs/
│   └── procedural_memory_design.md   # Design doc for the memory layer
├── prompts/
│   └── system_prompt.md   # Full prompt library documentation
├── .env                   # Secrets (DB password, LLM endpoint)
└── README.md              # This file
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--init-db` | Create/verify PostgreSQL tables |
| `--gen-personas N` | Generate N personas and insert into DB |
| `--generate` | Run question generation for all `pending` personas (resumable) |
| `--batch-size N` | Concurrent LLM calls per batch (default: 50) |
| `--migrate` | Apply additive schema migrations (status/attempts/dedup/memory columns) |
| `--report` | Print coverage + quality + diversity report |
| `--topic-report` | Generate `topic_report.html` (visual topic distribution) |
| `--test` | Micro-batch QA run (batch_size=100) |

### Examples

```bash
# Full pipeline
python main.py --init-db
python main.py --gen-personas 25000
python main.py --generate --batch-size 50

# Micro-batch for testing
python main.py --init-db
python main.py --gen-personas 100
python main.py --test
```

## The 7 Human-Like Guidelines

The system prompt enforces these rules to prevent AI-sounding output:

1. **Emotional Sequencing** — Start with frustration/confusion/urgency before asking
2. **Persona Consistency** — Vocabulary matches education level and region
3. **Register & Formatting** — Banned words list; mandate fragments, run-ons, lowercase
4. **Verbosity** — Question length tied to emotional state
5. **Pragmatic Coherence** — Implied context, no over-explaining
6. **Anti-Sycophancy** — Demanding/inquisitive/desperate tone, never overly polite
7. **Theory of Mind** — Assume chatbot knows only basics

## Procedural Memory

A GPU-backed memory layer keeps 25k questions from collapsing into repeated
phrasings. It adds **zero extra LLM calls** — only one local embedding per
question. See `docs/procedural_memory_design.md` for the full design.

- **Embedder:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, Bengali-capable),
  loaded on `cuda` by default (~0.5 GB VRAM; falls back to CPU automatically).
- **Style memory (anti-repetition):** stores each question's *opener* + embedding
  per `(profession, region)`. Before generating, overused openers are injected as
  a "do not start like this" constraint; after generating, a cosine near-duplicate
  check (`> MEMORY_SIM_THRESHOLD`) routes the draft into the rewrite loop.
- **Exemplar memory (few-shot):** high-scoring past questions from a *similar
  persona but a different topic* are injected as style examples (voice transfer,
  not content copying).
- Memory is **primed from the DB at run start** and persisted via the normal
  question insert, so it survives across runs. Toggle with `MEMORY_ENABLED`.

Diversity shows up in `--report` as the **distinct-opener ratio** (higher = more
varied phrasing) and in the run's completion line as the `near_dup` count.

## Quality Control (per question)

1. **Generate** draft (qwen3 with thinking disabled for speed).
2. **Gate** — Bengali-script ratio ≥ `GEN_MIN_BENGALI_RATIO` + banned-phrase check.
3. **Judge** — a *separate* LLM call returns `{verdict, is_bengali, reasons}` JSON
   (robustly parsed; recovers fields even from malformed JSON).
4. **Near-dup** — semantic check against procedural memory.
5. **Rewrite** up to `GEN_MAX_REWRITES` times; a **forceful anti-romanization
   escalation** fires when output comes back in Latin letters. The **best**
   attempt (ranked by pass-then-Bengali-ratio) is kept — never a worse rewrite.

## Database Schema

### `personas` table
| Column | Type | Description |
|--------|------|-------------|
| persona_id | SERIAL PK | Auto-generated ID |
| age | INT | Persona age (18-80) |
| gender | VARCHAR(50) | male / female / other |
| location | VARCHAR(100) | Bangladeshi district/region |
| profession | VARCHAR(100) | Occupation |
| social_status | VARCHAR(50) | Income bracket |
| backstory | TEXT | Generated narrative |
| json_metadata | JSONB | Full persona details (includes pain_point, education, random_seed) |
| processed | BOOLEAN | Legacy flag (kept in sync with `status`) |
| status | VARCHAR | `pending` / `done` / `failed` (drives the resumable loop) |
| attempts | INT | Retry count; at `GEN_MAX_ATTEMPTS` the persona is dead-lettered |
| error | TEXT | Last error for a failed persona |

### `hlq_questions` table
> Named `hlq_questions` (not `generated_questions`) because `gov_spider_db` is
> shared with another project that owns a differently-shaped `generated_questions`.

| Column | Type | Description |
|--------|------|-------------|
| question_id | SERIAL PK | Auto-generated ID |
| persona_id | INT FK | References personas(persona_id) |
| question_text | TEXT | Final generated question(s), Bengali |
| cot_log | TEXT | Full audit log incl. raw LLM response, gates, judge verdicts |
| random_seed | INT | Seed used for generation |
| dedup_hash | TEXT | Normalized hash for exact duplicate detection |
| quality_flags | JSONB | bengali_ratio, banned_phrases, judge verdict, duplicate/near_dup flags |
| embedding | BYTEA | float32[384] sentence embedding (procedural memory) |
| opener | TEXT | First ~6 tokens — phrasing fingerprint for anti-repetition |
| quality_score | REAL | judge PASS + bengali_ratio − dup penalty (exemplar ranking) |
| created_at | TIMESTAMP | Generation timestamp |

## Reproducibility Guide

- **Personas are fully reproducible** from the master seed — the same
  `persona_id`/`random_seed` always regenerates the identical persona.
- **Questions are only *approximately* reproducible.** vLLM continuous batching
  with `temperature > 0` is not deterministic from `seed` alone, so re-running a
  persona may yield a slightly different question. For exact auditability, the
  **raw LLM response is stored in `cot_log`** — you can always see precisely what
  was generated and why it passed/failed the gates and judge.

To re-generate a persona's question:
```sql
UPDATE personas SET status = 'pending', attempts = 0 WHERE persona_id = <ID>;
DELETE FROM hlq_questions WHERE persona_id = <ID>;
```
Then run `python main.py --generate --batch-size 1`.

## Monitoring

During long runs, monitor progress with:

```bash
# Coverage + quality + diversity report
python main.py --report

# Visual topic-distribution report → topic_report.html
python main.py --topic-report

# Watch CSV output in real-time
tail -f personas_questions.csv

# Check progress in PostgreSQL
psql -d gov_spider_db -c "SELECT status, COUNT(*) FROM personas GROUP BY status;"

# Check generation log
tail -f generation.log
```

## Configuration

All settings are loaded from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| DB_NAME | gov_spider_db | PostgreSQL database name |
| DB_USER | postgres | Database user |
| DB_PASSWORD | password | Database password |
| DB_HOST | localhost | Database host |
| DB_PORT | 5432 | Database port |
| LLM_URL | http://localhost:5000/v1/chat/completions | vLLM endpoint |
| LLM_MODEL | qwen3-35b-awq | Model name |
| GEN_CONCURRENCY | 50 | Concurrent in-flight LLM requests |
| DB_POOL_MAX | 20 | Max pooled DB connections |
| GEN_MAX_ATTEMPTS | 3 | Retries before a persona is dead-lettered (`failed`) |
| GEN_TEMPERATURE | 0.8 | Sampling temperature |
| GEN_MIN_BENGALI_RATIO | 0.65 | Min Bengali-script fraction to pass the language gate |
| GEN_MAX_REWRITES | 3 | Max in-cycle rewrites before keeping the best candidate |
| MEMORY_ENABLED | true | Enable the procedural-memory layer |
| MEMORY_MODEL | paraphrase-multilingual-MiniLM-L12-v2 | Sentence-embedding model |
| MEMORY_DEVICE | cuda | Embedder device (`cuda` / `cpu`) |
| MEMORY_SIM_THRESHOLD | 0.92 | Cosine ≥ this ⇒ near-duplicate ⇒ rewrite |
| MEMORY_K_EXEMPLARS | 3 | Few-shot exemplars injected per persona |
| MEMORY_AVOID_OPENERS_N | 5 | Overused openers warned against per cluster |
| MEMORY_MIN_EXEMPLAR_SCORE | 1.4 | Min quality_score to enter the exemplar bank |

> **GPU note:** the embedder shares the GPU with vLLM. On a 48 GB card with vLLM
> holding ~44 GB, the ~0.5 GB embedder still fits; set `MEMORY_DEVICE=cpu` if VRAM
> is tight.