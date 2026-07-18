# Two-Axis LLM Judge

Standalone, resumable quality-control judge over the generated questions in
`hlq_questions`. Uses the same vLLM (qwen3) endpoint and Postgres pool as the
parent project. **Evaluate + report only** — it never modifies generated data.

## Criteria

Each question is scored on two independent axes; **overall = FAIL if either fails**:

- **Axis 1 — Linguistic & Socio-Demographic Alignment:** does the vocabulary,
  register, and mobile-typing style match the persona? (rural/RMG → plain simple
  Bengali; tech/urban → natural Banglish). FAIL on "AI presence" — flawless/formal
  prose from a low-education persona, translated-template feel.
- **Axis 2 — Contextual & Pragmatic Correctness:** is the concern realistic for
  *this* persona, addressed naturally to a chatbot? FAIL on logical leaps or
  assumed tech/institutional knowledge the persona lacks.

Every **FAIL carries a concrete, specific reason** (stored + reported). A FAIL
returned without a reason is treated as a parse failure and re-judged.

## Usage

```bash
# Judge all unjudged questions (resumable — skips already-judged)
python judge/run_judge.py

# Limited / tuned
python judge/run_judge.py --limit 500 --batch-size 25

# Summary: pass rates per axis + worst cohorts + sample FAIL reasons
python judge/run_judge.py --report
```

Run it **after** generation finishes so it doesn't compete with vLLM.

## Config (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `JUDGE_CONCURRENCY` | 25 | Concurrent judge requests |
| `JUDGE_TEMPERATURE` | 0.1 | Low → consistent verdicts |
| `JUDGE_MAX_TOKENS` | 500 | Response budget |
| `JUDGE_THINKING` | false | qwen3 reasoning (on = slower, deeper; use when vLLM idle) |
| `JUDGE_TIMEOUT` | 120 | Per-request seconds |

## Output

- **Table `judge_evaluations`** — `question_id`, `persona_id`, `axis1_pass` +
  `axis1_reason`, `axis2_pass` + `axis2_reason`, `overall_pass`, `raw_response`,
  `model`. `overall_pass` is recomputed as `axis1_pass AND axis2_pass`.
- **CSV** `judge/judge_evaluations.csv` — persona fields + question + verdicts/reasons.

## Files

- `judge_config.py` — settings (reuses parent `LLM_CONFIG`/`DB_CONFIG`).
- `judge_prompt.py` — the two-axis system prompt + message builder.
- `judge_db.py` — table DDL, resumable fetch, save, CSV.
- `run_judge.py` — async engine, robust JSON verdict parser, `--report`.
