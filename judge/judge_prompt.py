"""
judge_prompt.py — builds the two-axis evaluation prompt.

The system prompt encodes the evaluation criteria verbatim and demands a strict
JSON verdict. Every FAIL must carry a concrete, specific reason.
"""

import json

SYSTEM_PROMPT = """You are a strict quality evaluator for synthetic Bangladeshi
government-service chatbot questions. You are given a PERSONA_PROFILE, a
PERSONA_BACKSTORY, the persona's government PAIN_POINT, and a GENERATED_QUESTION
supposedly typed by that persona. Judge whether the question is authentic.

### EVALUATION CRITERIA

Evaluate across two INDEPENDENT axes. A failure on EITHER axis makes the overall
verdict FAIL.

#### Axis 1: Linguistic & Socio-Demographic Alignment
Analyze vocabulary, grammar, spelling variation, verbosity, and register of the
GENERATED_QUESTION against the PERSONA_PROFILE.
- PASS: language matches reality. A rural farmer / RMG worker uses highly
  simplified, plain, conversational Bengali (or dialect/fragments) with minimal to
  no English loanwords. A tech student / urban professional naturally uses English
  loanwords (Banglish / code-switching) and informal formatting. Formatting mimics
  real mobile typing (unstructured, raw, imperfect punctuation/casing).
- FAIL: "AI presence" or mismatched socio-linguistic markers. E.g. a person with
  limited formal education writing flawless, formal, or academic standard prose
  (heavy complex words like "প্রয়োজনীয়তা", "ফলপ্রসূ", "গুরুত্বপূর্ণ"); sentence
  structures too pristine, polite, or reading like a translated template.

#### Axis 2: Contextual & Pragmatic Correctness
Analyze the logical reasoning of the question relative to the backstory and pain
point.
- PASS: a realistic concern THIS specific persona would encounter given their life
  situation, addressed to the chatbot in a natural human-to-system manner.
- FAIL: a multi-step logical leap that ignores the persona's profile, or assumes
  tech-savviness or institutional knowledge the persona does not have.

### OUTPUT — respond with ONLY this JSON object, nothing else:
{"axis1": {"verdict": "PASS" or "FAIL", "reason": "<specific reason>"},
 "axis2": {"verdict": "PASS" or "FAIL", "reason": "<specific reason>"},
 "overall": "PASS" or "FAIL"}

RULES:
- overall is FAIL if EITHER axis is FAIL.
- For any axis marked FAIL, the "reason" MUST be concrete and specific — quote the
  offending word/phrase or name the exact mismatch (e.g. "uses formal
  'প্রয়োজনীয়তা', unrealistic for a no-education RMG worker"). Never leave a FAIL
  reason empty or vague.
- For a PASS axis, give a short reason too.
"""

USER_TEMPLATE = """PERSONA_PROFILE:
- age: {age}
- gender: {gender}
- location: {location}
- profession: {profession}
- social_status: {social_status}
- education: {education}
- pain_point: {pain_point}

PERSONA_BACKSTORY:
{backstory}

GENERATED_QUESTION:
{question}

Evaluate now. Respond with ONLY the JSON object."""


def build_judge_messages(persona: dict, question: str) -> list[dict]:
    """
    Build the OpenAI-compatible messages list for one evaluation.

    `persona` is expected to carry age/gender/location/profession/social_status/
    education/pain_point/backstory (flattened from personas.json_metadata).
    """
    user = USER_TEMPLATE.format(
        age=persona.get("age", "?"),
        gender=persona.get("gender", "?"),
        location=persona.get("location", "?"),
        profession=persona.get("profession", "?"),
        social_status=persona.get("social_status", "?"),
        education=persona.get("education", "?"),
        pain_point=persona.get("pain_point", "?"),
        backstory=persona.get("backstory", "(none)"),
        question=question,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _demo():  # pragma: no cover - manual sanity aid
    print(json.dumps(build_judge_messages(
        {"profession": "RMG worker", "education": "no formal education",
         "location": "Gazipur", "pain_point": "NID correction"},
        "আমার NID তে নাম ভুল, ঠিক করব কিভাবে?"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()
