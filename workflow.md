# Workflow: Bangladeshi Persona & Human-Like Question Generation Agent

> Step-by-step implementation guide derived from `workplan.md`.

---

## Phase 1 — Environment Setup

### 1.1 Create the Python project structure

```
human_like_ques_generator_bd/
├── config.py              # DB + LLM settings
├── db.py                  # PostgreSQL helpers (connect, transact)
├── persona_generator.py   # Matrix-based persona creation
├── prompt_engine.py       # System prompt builder + 7 guidelines
├── cot_module.py          # Chain-of-Thought self-reflection
├── question_generator.py  # Core generation loop
├── main.py                # CLI entry-point (micro-batch / full run)
├── requirements.txt
├── prompts/
│   └── system_prompt.md   # Prompt library
├── .env                   # Secrets (DB password, etc.)
└── README.md
```

### 1.2 Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install openai psycopg2-binary pandas python-dotenv asyncio aiohttp
pip freeze > requirements.txt
```

### 1.3 Create `.env` file

```dotenv
DB_NAME=gov_spider_db
DB_USER=postgres
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432
LLM_URL=http://localhost:5000/v1/chat/completions
LLM_MODEL=qwen3-35b-awq
```

### 1.4 Create the database tables

Connect to PostgreSQL and run:

```sql
-- Personas table
CREATE TABLE IF NOT EXISTS personas (
    persona_id SERIAL PRIMARY KEY,
    age INT,
    gender VARCHAR(50),
    location VARCHAR(100),
    profession VARCHAR(100),
    social_status VARCHAR(50),
    backstory TEXT,
    json_metadata JSONB,
    processed BOOLEAN DEFAULT FALSE
);

-- Generated questions table
CREATE TABLE IF NOT EXISTS generated_questions (
    question_id SERIAL PRIMARY KEY,
    persona_id INT REFERENCES personas(persona_id),
    question_text TEXT,
    cot_log TEXT,
    random_seed INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 1.5 Initialise the CSV backup

Create `personas_questions.csv` with these headers:

```
persona_id,age,gender,location,profession,social_status,backstory,question_text,cot_log,random_seed,created_at
```

---

## Phase 2 — `config.py` — Centralised Configuration

```python
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "gov_spider_db"),
    "user":   os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "host":   os.getenv("DB_HOST", "localhost"),
    "port":   int(os.getenv("DB_PORT", 5432)),
}

LLM_CONFIG = {
    "url":   os.getenv("LLM_URL", "http://localhost:5000/v1/chat/completions"),
    "model": os.getenv("LLM_MODEL", "qwen3-35b-awq"),
}
```

---

## Phase 3 — `db.py` — Database Helpers

Implement the following functions:

| Function | Purpose |
|---|---|
| `get_connection()` | Return a `psycopg2` connection using `DB_CONFIG`. |
| `init_tables()` | Run the CREATE TABLE statements from Phase 1.4. |
| `insert_persona(conn, persona: dict)` | Transactional insert into `personas`. Returns `persona_id`. |
| `fetch_unprocessed_personas(conn, batch_size: int)` | `SELECT` personas where `processed = FALSE`, limited to `batch_size`. |
| `save_question(conn, persona_id, question_text, cot_log, random_seed)` | Transactional insert into `generated_questions`; set `personas.processed = TRUE`. Use `BEGIN`/`COMMIT`/`ROLLBACK`. |
| `append_to_csv(row: dict)` | Append a single row to `personas_questions.csv`. |

---

## Phase 4 — `persona_generator.py` — Matrix-Driven Persona Creation

### 4.1 Define weighted demographic arrays

```python
REGIONS = [
    "Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet",
    "Rangpur", "Barishal", "Mymensingh", "Kurigram", "Bhola",
    "Chittagong Hill Tracts", "Cox's Bazar", "Comilla", ...
]

PROFESSIONS = [
    "RMG worker", "expatriate worker's wife", "university student",
    "shrimp farmer", "rickshaw puller", "schoolteacher",
    "small shopkeeper", "day labourer", "tech startup employee",
    "government clerk", "nurse", "freelance web developer", ...
]

PAIN_POINTS = [
    "NID correction", "agricultural subsidies", "passport renewal",
    "land registration", "birth certificate", "education stipend",
    "electricity billing dispute", "voter ID issue", ...
]
```

### 4.2 Implement `generate_personas(count=25000)`

- Use `itertools.product` or weighted random sampling over the matrix arrays.
- Assign each persona a unique `random_seed` (e.g., `random.randint(0, 2**31)`).
- Build a JSON object per persona with keys: `age`, `gender`, `location`, `profession`, `social_status`, `backstory`, `json_metadata`.
- Bulk-insert all personas into the `personas` table via `db.insert_persona()`.

---

## Phase 5 — `prompt_engine.py` — System Prompt Builder

### 5.1 Build the system prompt template

Encode the **7 human-like guidelines** directly in the system prompt:

1. **Emotional Sequencing** — optionally start with frustration/confusion/urgency.
2. **Persona Consistency** — bind vocabulary to education level and region.
3. **Register & Formatting** — ban words like "delve", "crucial", "furthermore"; mandate sentence fragments, run-ons, lowercase.
4. **Verbosity** — tie question length to emotional state.
5. **Pragmatic Coherence** — allow implied context (e.g., "they misspelled my name again").
6. **Sycophancy/Stance** — tone must be demanding, inquisitive, or desperate; never overly polite.
7. **Theory of Mind** — assume the chatbot knows only basics; include phrases like "can you even help me with this?"

### 5.2 Implement `build_prompt(persona_json: dict) -> list[dict]`

- Return an OpenAI-compatible `messages` list:
  - `system`: The full system prompt with the 7 guidelines injected.
  - `user`: The persona JSON + instruction to generate 1-3 questions wrapped in `<draft_questions>` XML tags.

---

## Phase 6 — `cot_module.py` — Chain-of-Thought & Self-Reflection

### 6.1 Implement `build_cot_prompt(draft_questions: str, persona: dict) -> list[dict]`

Force the model to output a `<reflection>` block that answers:

- **Check 1:** "Does this sound like an AI wrote it? Overly formal transition words?"
- **Check 2:** "Is this exactly how a [profession] from [location] would type on a phone?"
- **Check 3:** "Is it too polite?"

### 6.2 Implement `validate_and_refine(llm_response: str) -> tuple[str, str]`

- Parse the `<reflection>` and `<draft_questions>` blocks.
- If reflection flags AI-sounding language → **re-prompt** the model within the same generation cycle with instructions to rewrite in a more raw/human tone.
- Return `(final_question_text, cot_log)`.

---

## Phase 7 — `question_generator.py` — Core Generation Loop

### 7.1 Implement `generate_for_batch(personas: list[dict])`

For each persona in the batch:

1. Apply the persona's `random_seed` to the LLM API call (as a seed parameter).
2. Call `prompt_engine.build_prompt(persona)`.
3. Send to the LLM endpoint (`LLM_CONFIG["url"]`) using `aiohttp`.
4. Route output to `cot_module.validate_and_refine()`.
5. On success, call `db.save_question()` and `db.append_to_csv()`.
6. On failure, log the error and continue to the next persona.

### 7.2 Implement `run(batch_size=50)`

```python
async def run(batch_size=50):
    conn = db.get_connection()
    while True:
        personas = db.fetch_unprocessed_personas(conn, batch_size)
        if not personas:
            break
        tasks = [generate_for_persona(p) for p in personas]
        await asyncio.gather(*tasks)
    conn.close()
```

- Use `asyncio.gather` with a semaphore (limit 50 concurrent calls) to avoid rate limits.
- The `processed` flag in the DB ensures safe pause/resume.

---

## Phase 8 — `main.py` — CLI Entry Point

```python
import argparse, asyncio
from db import init_tables
from persona_generator import generate_personas
from question_generator import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-db",       action="store_true", help="Create DB tables")
    parser.add_argument("--gen-personas",  type=int, default=0,  help="Generate N personas")
    parser.add_argument("--generate",      action="store_true", help="Run question generation")
    parser.add_argument("--batch-size",    type=int, default=50, help="Concurrent batch size")
    parser.add_argument("--test",          action="store_true", help="Micro-batch of 100 for QA")
    args = parser.parse_args()

    if args.init_db:
        init_tables()
    if args.gen_personas > 0:
        generate_personas(args.gen_personas)
    if args.test:
        asyncio.run(run(batch_size=100))   # micro-batch for QA
    elif args.generate:
        asyncio.run(run(batch_size=args.batch_size))

if __name__ == "__main__":
    main()
```

---

## Phase 9 — Testing & Quality Control

### 9.1 Micro-batch run (100 contrasting personas)

```bash
python main.py --init-db
python main.py --gen-personas 100
python main.py --test
```

### 9.2 Manual review checklist

- [ ] Open `personas_questions.csv` and audit for "AI tells" (perfect punctuation, unnatural context-setting).
- [ ] Verify regional vocabulary matches the persona's location.
- [ ] Check that banned words ("delve", "crucial", "furthermore") are absent.
- [ ] Confirm emotional tone is not overly polite.
- [ ] Validate that `random_seed` reproduces the same output when re-run.

### 9.3 Parameter tuning

- Adjust LLM `temperature` between **0.7 – 0.9** based on micro-batch results.
- Refine CoT reflection instructions if too many false positives/negatives.

---

## Phase 10 — Full-Scale Run & Monitoring

```bash
python main.py --gen-personas 25000
python main.py --generate --batch-size 50
```

### Monitoring during the run

- Periodically `tail -f personas_questions.csv` to check for mode collapse.
- Query progress: `SELECT COUNT(*) FROM personas WHERE processed = TRUE;`
- The script can be safely stopped and restarted — the `processed` flag prevents duplicates.

---

## Phase 11 — Documentation & Delivery

- [ ] Comment all Python scripts (especially batching logic and DB pooling).
- [ ] Finalise `prompts/system_prompt.md` with the exact prompts used for generation and reflection.
- [ ] Write a **Reproducibility Guide** section in `README.md` explaining how to take a `persona_id` + `random_seed` and regenerate the exact same question.

---

## Quick-Reference: Execution Order

| Step | Command / Action | Purpose |
|------|-----------------|---------|
| 1 | `pip install -r requirements.txt` | Install deps |
| 2 | Create `.env` | Configure credentials |
| 3 | `python main.py --init-db` | Create DB tables |
| 4 | `python main.py --gen-personas 100` | Seed test personas |
| 5 | `python main.py --test` | Micro-batch QA run |
| 6 | Review `personas_questions.csv` | Manual audit |
| 7 | Tune temperature / CoT prompts | Iterate |
| 8 | `python main.py --gen-personas 25000` | Full persona gen |
| 9 | `python main.py --generate --batch-size 50` | Full question gen |
| 10 | Monitor CSV + DB | Watch for drift |
| 11 | Document everything | Final delivery |
