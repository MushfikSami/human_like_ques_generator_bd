"""
cot_module.py — Judge, Quality Checks & Deduplication

Anti-AI-tone enforcement, redesigned so grading is actually meaningful:

  1. `parse_draft`            — pull the questions out of the generator output.
  2. `programmatic_checks`    — deterministic gates: Bengali-script ratio +
                                a trimmed, Bengali-aware banned-phrase list.
  3. `build_cot_prompt`       — a SEPARATE judge LLM call (not self-grading in
                                the same completion) that returns structured
                                JSON {verdict, reasons, is_bengali}.
  4. `parse_judge`            — parse that JSON robustly.
  5. `build_rewrite_prompt`   — second-pass rewrite when a gate/judge fails.
  6. `dedup_hash`             — normalized hash for near-duplicate detection.

The old approach (a) had the model grade itself in the same call — which almost
always self-reported PASS — and (b) banned English words that never appear in
Bengali output. Both are fixed here.
"""

import re
import json
import hashlib
import logging
import unicodedata

logger = logging.getLogger(__name__)

# Unicode Bengali block: U+0980–U+09FF.
_BENGALI_RE = re.compile(r"[ঀ-৿]")
# "Word-ish" characters we count toward the language ratio denominator
# (Bengali + Latin letters + digits); whitespace/punctuation are ignored.
_ALNUM_RE = re.compile(r"[ঀ-৿A-Za-z0-9]")

# Bengali formal/officialese phrases that betray AI/bureaucratic register.
# These DO occur in Bengali output, unlike the old English list.
BANNED_BENGALI_PHRASES = [
    "অনুগ্রহপূর্বক", "এতদ্বারা", "উক্ত বিষয়ে", "মহোদয়", "সম্মানিত",
    "বিনীত অনুরোধ", "সদয় অবগতি", "প্রসঙ্গে জানানো যাচ্ছে", "স্মারকে",
]
# A few English tells that would look wrong inside a phone-typed Bengali message.
BANNED_ENGLISH_PHRASES = [
    "i would like to inquire", "i hope this message finds you well",
    "kindly", "furthermore", "please be advised",
]


# ─── Parsing ─────────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Reasoning-model preambles that sometimes leak when the model skips the tags.
_REASONING_PREAMBLE_RE = re.compile(
    r"^(here'?s?\s+(a|my)\s+thinking\s+process|let me think|okay,?\s+so).*",
    re.IGNORECASE | re.DOTALL,
)


def parse_draft(llm_response: str) -> str:
    """
    Extract the questions from a generator response.

    Strips reasoning-model `<think>` blocks first, then uses the LAST
    <draft_questions> block if present. If no tags are found, returns an empty
    string rather than leaking the model's reasoning monologue — the caller's
    quality gate will then correctly fail and trigger a rewrite.
    """
    cleaned = _THINK_RE.sub("", llm_response).strip()

    all_drafts = re.findall(r"<draft_questions>(.*?)</draft_questions>",
                            cleaned, re.DOTALL)
    if all_drafts:
        return all_drafts[-1].strip()

    # No tags. If the response is clearly a reasoning monologue, drop it so it
    # doesn't get persisted as the "question".
    if _REASONING_PREAMBLE_RE.match(cleaned):
        logger.warning("No <draft_questions> tags and response looks like reasoning; discarding.")
        return ""

    logger.warning("No <draft_questions> tags found; using cleaned response.")
    return cleaned


# ─── Deterministic Quality Gates ─────────────────────────────────────────────

def bengali_ratio(text: str) -> float:
    """Fraction of alphanumeric characters that are Bengali script (0.0-1.0)."""
    alnum = _ALNUM_RE.findall(text)
    if not alnum:
        return 0.0
    bengali = _BENGALI_RE.findall(text)
    return len(bengali) / len(alnum)


def _find_banned(text: str) -> list[str]:
    lower = text.lower()
    found = [p for p in BANNED_ENGLISH_PHRASES if p in lower]
    found += [p for p in BANNED_BENGALI_PHRASES if p in text]
    return found


def programmatic_checks(text: str, min_bengali_ratio: float) -> tuple[bool, dict]:
    """
    Run deterministic quality gates on the question text.

    Returns:
        (ok, flags) where ok is True if all gates pass and flags is a dict
        recording the measured values / violations for auditing.
    """
    ratio = bengali_ratio(text)
    banned = _find_banned(text)
    empty = len(text.strip()) == 0

    ok = (not empty) and ratio >= min_bengali_ratio and not banned
    flags = {
        "bengali_ratio": round(ratio, 3),
        "banned_phrases": banned,
        "empty": empty,
        "programmatic_pass": ok,
    }
    return ok, flags


# ─── Judge Call (separate LLM completion) ────────────────────────────────────

def _persona_fields(persona: dict) -> tuple[str, str, str]:
    if isinstance(persona.get("json_metadata"), dict):
        meta = persona["json_metadata"]
    elif isinstance(persona.get("json_metadata"), str):
        meta = json.loads(persona["json_metadata"])
    else:
        meta = persona or {}
    return (
        meta.get("profession", (persona or {}).get("profession", "unknown")),
        meta.get("location", (persona or {}).get("location", "unknown")),
        meta.get("education", "unknown"),
    )


def build_cot_prompt(draft_questions: str, persona: dict) -> list[dict]:
    """
    Build a SEPARATE judge prompt that scores the draft and returns JSON.

    A distinct completion (rather than self-grading inside the generator call)
    makes the verdict far less likely to be a rubber-stamp PASS.
    """
    profession, location, education = _persona_fields(persona)

    system_content = (
        "You are a strict reviewer of Bengali (বাংলা) chatbot questions supposedly "
        "typed by real Bangladeshi citizens on their phones. Judge authenticity "
        "harshly — most AI-written text is too clean, too polite, and too "
        "well-structured. Respond with ONLY a JSON object, no prose."
    )

    user_content = f"""Persona: a {profession} from {location} with {education} education.

Draft questions:
{draft_questions}

Judge whether these read as authentically typed by THIS person on a phone.
Fail if: overly formal/polite, unnaturally structured, perfect grammar, sounds
AI-written, or not genuinely in Bengali script.

Return ONLY this JSON:
{{"verdict": "PASS" or "FAIL", "is_bengali": true or false, "reasons": ["..."]}}

/no_think"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_VERDICT_RE = re.compile(r"verdict\W{0,4}(PASS|FAIL)", re.IGNORECASE)
_ISBENGALI_RE = re.compile(r"is_bengali\W{0,4}(true|false)", re.IGNORECASE)


def _iter_brace_objects(text: str):
    """Yield balanced-brace {...} substrings, so we don't greedily merge two."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _coerce_result(data: dict) -> dict:
    verdict = str(data.get("verdict", "PASS")).upper()
    return {
        "verdict": "FAIL" if verdict == "FAIL" else "PASS",
        "is_bengali": bool(data.get("is_bengali", True)),
        "reasons": data.get("reasons", []) or [],
    }


def parse_judge(judge_response: str) -> dict:
    """
    Parse the judge's verdict robustly.

    Strategy (each falls through to the next):
      1. Strip <think> blocks and markdown code fences.
      2. Try json.loads on each balanced {...} object found.
      3. Regex-recover `verdict` / `is_bengali` directly from the text, so a
         malformed-JSON reply still yields a real verdict instead of a blind PASS.
      4. Only if nothing at all is found, default to a lenient PASS (the
         deterministic programmatic gates still apply either way).

    Returns a dict with keys: verdict (PASS/FAIL), is_bengali (bool), reasons (list).
    """
    text = _THINK_RE.sub("", judge_response or "").strip()

    # Prefer content inside a ```json ... ``` fence if present.
    fence = _FENCE_RE.search(text)
    scan = fence.group(1) if fence else text

    # 2. Try real JSON on each balanced object.
    for candidate in _iter_brace_objects(scan):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return _coerce_result(data)
        except json.JSONDecodeError:
            continue

    # 3. Regex recovery — pull the fields out even if the JSON is malformed.
    verdict_m = _VERDICT_RE.search(text)
    bengali_m = _ISBENGALI_RE.search(text)
    if verdict_m or bengali_m:
        logger.warning("Judge JSON unparseable; recovered fields via regex.")
        return {
            "verdict": verdict_m.group(1).upper() if verdict_m else "PASS",
            "is_bengali": (bengali_m.group(1).lower() == "true") if bengali_m else True,
            "reasons": ["recovered from malformed judge json"],
        }

    # 4. Nothing usable.
    logger.warning("Judge returned no usable verdict; defaulting to PASS.")
    return {"verdict": "PASS", "is_bengali": True, "reasons": ["unparseable judge"]}


# ─── Rewrite Prompt ──────────────────────────────────────────────────────────

REWRITE_PROMPT_TEMPLATE = """The previous draft was rejected.

PROBLEMS:
{problems}

ORIGINAL DRAFT:
{draft}

PERSONA: {profession} from {location}, {education} education.

Rewrite the questions so they read like THIS person really typed them on a phone:
1. ALL questions in Bengali (বাংলা) script — NOT transliteration
2. Raw, informal, imperfect — fragments and run-ons are good
3. Regional/dialect flavour where it fits; not polite or formal
4. English words like NID, online, SMS, app may be mixed in

Output ONLY the rewritten questions inside <draft_questions> tags.

/no_think"""


def build_rewrite_prompt(draft_questions: str, problems: str, persona: dict) -> list[dict]:
    """Build a rewrite prompt when a draft fails the gates/judge."""
    profession, location, education = _persona_fields(persona)
    user_content = REWRITE_PROMPT_TEMPLATE.format(
        problems=problems, draft=draft_questions,
        profession=profession, location=location, education=education,
    )
    return [
        {"role": "system",
         "content": "You rewrite text to sound like a real person typed it on a "
                    "phone in Bengali (বাংলা). Output ONLY <draft_questions> tags."},
        {"role": "user", "content": user_content},
    ]


# ─── Deduplication ───────────────────────────────────────────────────────────

def dedup_hash(text: str) -> str:
    """
    Normalized hash for near-duplicate detection.

    Lowercases, strips punctuation/whitespace runs and Unicode-normalizes so
    trivially different phrasings of the same question collide.
    """
    norm = unicodedata.normalize("NFKC", text).lower()
    norm = re.sub(r"[^ঀ-৿a-z0-9]+", " ", norm).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()
