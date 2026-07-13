# Bangladeshi Persona & Human-Like Question Generation Agent

A pipeline for generating 25,000 diverse Bangladeshi personas and producing
human-like Bengali questions about government services using a local LLM
(Qwen3-35B via vLLM). Quality is enforced by deterministic gates (Bengali-script
ratio + banned-phrase list) plus a **separate LLM judge call**, with a reliable,
resumable batch engine. See `workflow.md` for the full architecture.

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
├── config.py              # DB + LLM settings (from .env)
├── db.py                  # PostgreSQL helpers (connect, transact, CSV backup)
├── persona_generator.py   # Matrix-based persona creation (60+ regions, 45+ professions)
├── prompt_engine.py       # System prompt builder with 7 human-like guidelines
├── cot_module.py          # Quality gates, LLM judge, rewrite loop, dedup
├── question_generator.py  # Async engine: pool, dead-letter, single-writer queue
├── report.py              # Coverage + quality report (--report)
├── main.py                # CLI entry-point
├── requirements.txt       # Python dependencies
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
| `--migrate` | Apply additive schema migrations (status/attempts/dedup columns) |
| `--report` | Print coverage + quality report |
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
| dedup_hash | TEXT | Normalized hash for near-duplicate detection |
| quality_flags | JSONB | bengali_ratio, banned_phrases, judge verdict, duplicate flag |
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
# Coverage + quality report
python main.py --report

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
| GEN_MIN_BENGALI_RATIO | 0.55 | Min Bengali-script fraction to pass the language gate |