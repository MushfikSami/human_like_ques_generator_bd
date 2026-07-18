# Design: Procedural Memory for the Question Generator

Status: approved, in implementation
Scope: **procedural memory** (style memory + exemplar memory). Local
`sentence-transformers` embedder on the **RTX A6000 (CUDA)**.
Out of scope for now: KB/RAG grounding, multi-turn conversation sessions.

---

## 1. Motivation

Today every question is a **stateless, single-shot LLM call**. Each call knows
only what [prompt_engine.py](../prompt_engine.py) `build_prompt` injects (the
persona) and nothing about the ~25k other questions. There is no memory across
calls except an exact-match `dedup_hash` computed in Python
([cot_module.py](../cot_module.py)). At scale this invites **mode collapse**:
repeated openers ("আবারও ঝামেলা…"), repeated sentence shapes, and no compounding
of what "good" looked like earlier.

**Procedural memory** gives the generator persistent state it reads before, and
writes after, each generation — without changing the model weights and without
adding any extra LLM calls.

## 2. Two subsystems

### 2.1 Style memory — anti-repetition (diversity)
- After each accepted question, persist its **opener** (first ~6 tokens) and its
  **embedding**, tagged by `(profession, region)`.
- **Before** generating, look up the most-overused openers for that persona's
  `(profession, region)` and inject a negative constraint:
  *"Do NOT start like these (already overused): …"*.
- **After** generating, cosine-compare the draft to recent same-cluster
  embeddings. If `sim > sim_threshold`, it is a **near-duplicate** → route into
  the existing rewrite loop (a strict upgrade over exact-hash dedup).

### 2.2 Exemplar memory — few-shot (quality compounds)
- Maintain a bank of **high-quality** past questions (judge PASS, high
  `bengali_ratio`, not a duplicate) with a scalar `quality_score`.
- For a new persona, retrieve *k* nearest exemplars **from a similar persona but a
  different topic**, and inject them as `## STYLE EXAMPLES (match tone, not
  content)`. Retrieving *same-topic* exemplars is intentionally avoided — it
  causes content copying; we want voice/register transfer only.

The two pull in opposite directions on purpose: exemplars → consistency of voice,
style memory → variety of phrasing. Together: *diverse but on-voice*.

## 3. Embedder

- Model: **`paraphrase-multilingual-MiniLM-L12-v2`** (384-dim, multilingual incl.
  Bengali, ~0.5 GB VRAM).
- Device: **`cuda`** (RTX A6000). Note: vLLM already holds ~44/48 GB; MiniLM fits
  in the ~4.8 GB free. `MEMORY_CONFIG["device"]` allows `cpu` fallback and the
  loader degrades to CPU if CUDA is unavailable or OOMs.
- Encoding runs via `asyncio.to_thread` so it overlaps with in-flight LLM
  requests. Torch releases the GIL during encode; a lock is added only if
  profiling shows contention.

## 4. Storage (Postgres, additive — same pattern as `hlq_*`)

New columns on `hlq_questions` (nullable, backfillable):

| Column | Type | Purpose |
|---|---|---|
| `embedding` | `BYTEA` | `np.float32[384].tobytes()` |
| `opener` | `TEXT` | first ~6 tokens, for opener-frequency lookups |
| `quality_score` | `REAL` | derived from `quality_flags` (judge + bengali_ratio − dup) |

**No `pgvector`.** Cosine similarity is done in-process with numpy — a 25k × 384
matrix is trivial. Embeddings persist as bytes so memory survives across runs:
`MemoryStore.prime(conn)` loads them into an in-RAM matrix at start.

## 5. Modules & API

### `memory_store.py` (new)
```python
class MemoryStore:
    def __init__(self, cfg): ...                 # lazy-loads embedder
    def prime(self, conn): ...                   # load embeddings/openers from DB
    def embed(self, text) -> np.ndarray: ...     # float32[384], normalized
    def get_context(self, persona) -> dict:      # {"avoid_openers":[...], "exemplars":[...]}
    def is_near_dup(self, vec, persona) -> bool
    def record(self, persona, text, vec, score): # update in-RAM index + opener counts
```
- In-RAM state: `emb_matrix` (np.ndarray), per-`(profession,region)` opener
  Counters, and an exemplar list kept sorted by `quality_score`.
- `record` is called **only from the single writer task**, so index mutation is
  serial — no new locks/races. DB persistence of the embedding/opener happens in
  the same writer transaction as `save_question`.

### Touch points
- **[config.py](../config.py)** — `MEMORY_CONFIG`: `enabled`, `model_name`,
  `device`, `sim_threshold`, `k_exemplars`, `avoid_openers_n`, `min_exemplar_score`.
- **[db.py](../db.py)** — migration for the 3 columns; `save_question` extended to
  write `embedding`/`opener`/`quality_score`; a `fetch_memory_rows` loader.
- **[prompt_engine.py](../prompt_engine.py)** — `build_prompt(persona,
  memory_context=None)`; renders the avoid-openers + exemplar blocks when present.
  `None` ⇒ exactly today's behavior (backward compatible).
- **[question_generator.py](../question_generator.py)** — build `MemoryStore`,
  `prime` at run start; per persona fetch `get_context`, embed the final text in
  the worker (`to_thread`), pass the vector through the queue item; the writer
  calls `record`. Near-dup check hooks into the existing rewrite loop.
- **[report.py](../report.py)** — diversity metrics: **distinct-opener ratio** and
  **mean pairwise cosine** (sampled), before/after.

## 6. `quality_score` definition
```
score = (1.0 if judge.verdict == "PASS" else 0.0)
      + bengali_ratio            # 0..1
      - (0.5 if duplicate else 0)
```
Only questions with `score >= min_exemplar_score` (default ~1.4) enter the
exemplar bank.

## 7. Concurrency & failure model (unchanged guarantees)
- Embedding happens in parallel workers (`to_thread`); **index writes stay in the
  single writer task**, preserving the "no CSV/DB races" property.
- If the embedder fails to load, `MEMORY_CONFIG["enabled"]` is treated as `False`
  for the run and generation proceeds exactly as today (graceful degradation).
- Dead-letter / resume logic is untouched.

## 8. Cost / performance
- **0 extra LLM calls.** One local GPU embedding per question (~1–5 ms on A6000),
  overlapped with LLM latency (~2 s). Negligible.
- VRAM: ~0.5 GB; monitor against vLLM's 44 GB. CPU fallback available.

## 9. Phasing
1. **Store + schema + config** — `memory_store.py`, migration, `MEMORY_CONFIG`; no
   behavior change yet (feature flag off).
2. **Style memory** — avoid-openers injection + near-dup → rewrite.
3. **Exemplar memory** — few-shot retrieval + injection.
4. **Diversity metrics** — report distinct-opener ratio & mean pairwise cosine.

## 10. Verification
- Unit: `record`→`get_context` round-trip; cosine correctness; opener extraction;
  embedder loads on CUDA.
- Integration on a **fresh 200-persona scratch table**: confirm avoid-openers and
  exemplars appear in prompts; near-dups trigger rewrites; embeddings persist and
  reload via `prime`.
- **Metric proof:** distinct-opener ratio ↑ and mean pairwise cosine ↓ vs. a
  no-memory control run over the same personas.
- The existing 25k data is untouched; new runs use a fresh table.
```
