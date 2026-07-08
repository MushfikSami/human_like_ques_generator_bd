Here is the comprehensive, step-by-step work plan for developing the persona and question generation agent. This blueprint is structured to be directly actionable for development, ensuring strict adherence to the human-like conversational constraints and the technical architecture you outlined.

---

### **Work Plan: Bangladeshi Persona and Human-Like Question Generation Agent**

#### **1. Environment Setup**

The foundation requires a robust Python environment capable of handling batch LLM requests and reliable database transactions.

* **Dependencies:** Initialize the environment with `openai` (or equivalent SDK for local models), `psycopg2-binary` (for PostgreSQL), `pandas` (for CSV handling), `python-dotenv` (for secrets), and `asyncio` (for concurrent batching).
* **Database Architecture:** Set up PostgreSQL with a relational schema to ensure data integrity.
```sql
CREATE TABLE personas (
    persona_id SERIAL PRIMARY KEY,
    age INT,
    gender VARCHAR(50),
    location VARCHAR(100),
    profession VARCHAR(100),
    social_status VARCHAR(50),
    backstory TEXT,
    json_metadata JSONB
);

CREATE TABLE generated_questions (
    question_id SERIAL PRIMARY KEY,
    persona_id INT REFERENCES personas(persona_id),
    question_text TEXT,
    cot_log TEXT,
    random_seed INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```


* **Local Logging:** Initialize `personas_questions.csv` with headers matching the database schema to serve as a local backup and quick-access analytical tool.

#### **2. Data Collection & Persona Generation**

To reach 25,000 diverse profiles, the persona generation must be matrix-driven rather than purely random to ensure comprehensive demographic coverage of Bangladesh.

* **Matrix Creation:** Define weighted arrays for regions (e.g., Dhaka, Kurigram, Bhola, Chittagong Hill Tracts), professions (e.g., RMG worker, expatriate worker's wife, university student, shrimp farmer), and specific government-related pain points (e.g., NID correction, agricultural subsidies, passport renewal).
* **JSON Structuring:** Programmatically generate 25,000 JSON objects using the matrix.
* **Seed Integration:** Assign a unique `random_seed` to each generated persona at this stage to lock in their specific traits and ensure reproducibility during the question generation phase.

#### **3. Prompt Engineering & Persona Initialization**

The system prompt is the critical control mechanism for bypassing standard AI conversational patterns.

* **Prompt Construction:** The prompt must inject the persona JSON and explicitly define the 7 human-like guidelines:
* *Emotional Sequencing:* Instruct the model to optionally start with a brief expression of frustration, confusion, or urgency before asking the question.
* *Persona Consistency:* Bind the vocabulary to the persona's education and region (e.g., a farmer might use localized terms for land, while a tech worker uses English loanwords).
* *Register and Formatting:* Ban words like "delve," "crucial," or "furthermore." Mandate sentence fragments, run-on sentences, and lowercase text where appropriate to mimic mobile typing.
* *Verbosity:* Tie question length to the persona's emotional state.
* *Pragmatic Coherence:* Allow for implied context (e.g., "they misspelled my name again" instead of "the election commission misspelled my name on my National Identity Card").
* *Sycophancy/Stance:* Ensure the tone is demanding, inquisitive, or desperate, not overly polite or accommodating to the AI.
* *Theory of Mind:* The persona should assume the chatbot only knows basic things, often asking "can you even help me with this?"



#### **4. Question Generation Loop**

This is the core execution script, designed to process the 25,000 personas efficiently.

* **Execution Flow:**
1. Fetch a batch of unprocessed personas from the database.
2. For each persona, apply their assigned `random_seed` to the LLM API call.
3. Feed the engineered prompt + persona JSON to the LLM, requesting 1-3 questions in a structured format (e.g., a specific XML tag `<draft_questions>`).
4. Route the output to the Chain-of-Thought (CoT) module.
5. Upon successful validation, execute a `psycopg2` transaction to insert the data into PostgreSQL and append it to the CSV.



#### **5. Chain-of-Thought & Self-Reflection Module**

To strictly enforce the anti-AI tone, the model must grade its own output before finalizing it.

* **The CoT Prompt:** Force the model to output a `<reflection>` block before the final question.
* *Check 1:* "Does this sound like an AI wrote it? Are there overly formal transition words?"
* *Check 2:* "Is this exactly how a [insert profession] from [insert location] would type this on a phone?"
* *Check 3:* "Is it too polite?"


* **Regeneration Trigger:** If the reflection block identifies standard AI phrasing, the prompt must instruct the model to discard the draft, adjust the tone to be more raw and human, and try again within the same generation cycle.

#### **6. Data Storage & Logging**

Data integrity across a 25,000-record run requires robust failure handling.

* **Transactional Inserts:** Use `BEGIN` and `COMMIT` blocks in `psycopg2`. If an insert fails, use `ROLLBACK` to prevent partial data corruption.
* **Metadata Logging:** Log the `persona_id`, the exact prompt used, the `cot_log` (the reflection text), the final `question_text`, and the `random_seed`. This allows developers to audit exactly *why* a specific question was generated.

#### **7. Testing & Quality Control**

Before running the full 25,000 batch, the pipeline must be validated.

* **Micro-Batching:** Run a test batch of 100 highly contrasting personas (e.g., a diplomat vs. a rural fisherman).
* **Human Review:** Manually audit the CSV output. Look for "AI tells" (e.g., perfect punctuation, unnatural context-setting).
* **Parameter Tuning:** Adjust the temperature (e.g., setting it to 0.7 - 0.9 for higher conversational variance) and refine the CoT instructions based on the micro-batch results.

#### **8. Documentation & Delivery**

Ensure the project is easily maintainable and reproducible.

* **Code Documentation:** Comment all Python scripts, specifically the batching logic and the DB connection pooling.
* **Prompt Library:** Maintain a markdown file of the exact prompts used for generation and reflection.
* **Reproducibility Guide:** Document how a developer can take a `persona_id` and `random_seed` from the database and regenerate the exact same question for debugging purposes.

#### **9. Final Deployment & Maintenance**

Execute the full scale generation and monitor the system.

* **Asynchronous Batching:** Use `asyncio` to run concurrent API calls (e.g., 50 at a time) to process the 25,000 personas efficiently without hitting rate limits.
* **Resumption Logic:** Implement a tracker in the database (e.g., a `processed` boolean column in the `personas` table) so the script can be safely paused and restarted without duplicating work.
* **Periodic Audits:** During the long run, periodically tail the CSV file to ensure the API hasn't drifted into repetitive patterns (mode collapse).


The db config:
"dbname": "gov_spider_db",
            "user": "postgres",
            "password": "password", # Update this if needed
            "host": "localhost",
            "port": 5432

the vllm config:
llm_url="http://localhost:5000/v1/chat/completions"
"model": "qwen3-35b-awq"  

