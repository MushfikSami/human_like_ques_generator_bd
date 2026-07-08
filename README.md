# Bangladeshi Persona & Human-Like Question Generation Agent

A pipeline for generating 25,000 diverse Bangladeshi personas and producing
human-like questions about government services using a local LLM (Qwen3-35B
via vLLM) with Chain-of-Thought self-reflection to ensure authenticity.

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
├── cot_module.py          # Chain-of-Thought self-reflection & rewrite loop
├── question_generator.py  # Core async generation loop (aiohttp + asyncio)
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
| `--generate` | Run question generation for all unprocessed personas |
| `--batch-size N` | Concurrent LLM calls per batch (default: 50) |
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
| processed | BOOLEAN | Whether questions have been generated |

### `generated_questions` table
| Column | Type | Description |
|--------|------|-------------|
| question_id | SERIAL PK | Auto-generated ID |
| persona_id | INT FK | References personas(persona_id) |
| question_text | TEXT | Final generated question(s) |
| cot_log | TEXT | Chain-of-Thought reflection log |
| random_seed | INT | Seed used for reproducibility |
| created_at | TIMESTAMP | Generation timestamp |

## Reproducibility Guide

To regenerate the exact same question for a specific persona:

1. Look up the `persona_id` and `random_seed` from the database:
   ```sql
   SELECT persona_id, json_metadata->>'random_seed' as seed
   FROM personas WHERE persona_id = <ID>;
   ```

2. The `random_seed` is passed to the LLM API as the `seed` parameter. Given the same:
   - Persona JSON (deterministic from the seed)
   - System prompt (version-controlled in `prompts/system_prompt.md`)
   - LLM model and temperature settings
   
   The output should be identical.

3. To verify: reset the persona's `processed` flag and re-run:
   ```sql
   UPDATE personas SET processed = FALSE WHERE persona_id = <ID>;
   DELETE FROM generated_questions WHERE persona_id = <ID>;
   ```
   Then run `python main.py --generate --batch-size 1`.

## Monitoring

During long runs, monitor progress with:

```bash
# Watch CSV output in real-time
tail -f personas_questions.csv

# Check progress in PostgreSQL
psql -d gov_spider_db -c "SELECT COUNT(*) as done FROM personas WHERE processed = TRUE;"
psql -d gov_spider_db -c "SELECT COUNT(*) as remaining FROM personas WHERE processed = FALSE;"

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