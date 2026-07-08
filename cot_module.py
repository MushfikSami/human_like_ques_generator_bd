"""
cot_module.py — Chain-of-Thought & Self-Reflection

Implements the anti-AI-tone enforcement layer. After initial question
generation, this module:
  1. Parses the <draft_questions> and <reflection> XML blocks from the LLM output.
  2. Evaluates the reflection verdict.
  3. If the reflection flags AI-sounding language, builds a rewrite prompt
     for a second pass within the same generation cycle.
  4. Returns the final question text and full CoT log for auditing.
"""

import re
import json
import logging

logger = logging.getLogger(__name__)

# ─── Banned Words List ───────────────────────────────────────────────────────
# If any of these appear in the generated questions, the output is flagged
# as AI-sounding regardless of the model's own reflection verdict.

BANNED_WORDS = [
    "delve", "crucial", "furthermore", "comprehensive", "facilitate",
    "in order to", "it is important to note", "i would like to inquire",
    "i hope this message finds you well", "pertaining to", "subsequently",
    "utilize", "regarding", "hence", "therefore", "nevertheless",
    "accordingly", "moreover", "whilst",
]

# ─── XML Parsing ─────────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> str | None:
    """
    Extract content between <tag> and </tag> from text.

    Args:
        text: The full LLM response text.
        tag: The XML tag name (without angle brackets).

    Returns:
        The content between the tags, or None if not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _contains_banned_words(text: str) -> list[str]:
    """
    Check if the text contains any banned AI-sounding words/phrases.

    Args:
        text: The question text to check.

    Returns:
        List of banned words found (empty if clean).
    """
    text_lower = text.lower()
    found = []
    for word in BANNED_WORDS:
        if word in text_lower:
            found.append(word)
    return found


# ─── CoT Prompt Builder ─────────────────────────────────────────────────────

REWRITE_PROMPT_TEMPLATE = """The previous draft questions were flagged as not meeting quality standards.

PROBLEMS IDENTIFIED:
{problems}

ORIGINAL DRAFT:
{draft}

PERSONA CONTEXT:
- Profession: {profession}
- Location: {location}
- Education: {education}

REWRITE INSTRUCTIONS:
1. ALL questions MUST be in Bengali (বাংলা) script — NOT English transliteration
2. Make it sound MUCH more human and raw
3. Use informal Bengali as this persona would actually type on their phone
4. Use regional dialect words where appropriate
5. Remove ANY formal phrasing completely
6. Some English words like NID, online, SMS, app are OK to mix in
7. Think: "Would this person REALLY type this on their phone in Bengali?"

Generate the rewritten questions in <draft_questions> tags:"""


def build_cot_prompt(draft_questions: str, persona: dict) -> list[dict]:
    """
    Build a Chain-of-Thought reflection prompt for evaluating draft questions.

    Forces the model to output a <reflection> block that checks:
    - Check 1: AI-sounding language detection
    - Check 2: Profession/location authenticity match
    - Check 3: Politeness level check

    Args:
        draft_questions: The raw draft questions text from the initial generation.
        persona: Dict with persona details (profession, location, education, etc.).

    Returns:
        OpenAI-compatible messages list for the reflection call.
    """
    # Extract persona details
    if "json_metadata" in persona and isinstance(persona["json_metadata"], dict):
        metadata = persona["json_metadata"]
    elif "json_metadata" in persona and isinstance(persona["json_metadata"], str):
        metadata = json.loads(persona["json_metadata"])
    else:
        metadata = persona

    profession = metadata.get("profession", persona.get("profession", "unknown"))
    location = metadata.get("location", persona.get("location", "unknown"))
    education = metadata.get("education", "unknown")

    system_content = """You are a quality checker for human-like question generation in Bengali (বাংলা). Your job is to evaluate whether generated questions sound authentically human or like AI-generated text, and whether they are properly written in Bengali script.

You MUST output a <reflection> block with these exact checks:

<reflection>
Check 1 — AI Detection: Does this sound like an AI wrote it? Are there overly formal transition words, perfect grammar, or unnaturally structured sentences?
Check 2 — Persona Match: Is this EXACTLY how a {profession} from {location} with {education} education would type this on a phone? Would they really use these words?
Check 3 — Politeness Check: Is it too polite? Does the tone match real frustrated/confused users?
Check 4 — Language Check: Is the output in Bengali (বাংলা) script? NOT English transliteration? Some English words like NID, online, SMS are acceptable.
Verdict: [PASS / FAIL]
Reason: [explain if FAIL]
</reflection>

If FAIL, rewrite the questions in Bengali (বাংলা) with a more raw, human tone in <draft_questions> tags.""".format(
        profession=profession, location=location, education=education
    )

    user_content = f"""Evaluate these draft questions for authenticity and Bengali language compliance:

<draft_questions>
{draft_questions}
</draft_questions>

This was written for a persona who is a {profession} from {location} with {education} education.

Check if it sounds human enough AND is properly in Bengali (বাংলা) script. Output your <reflection> and if needed, a rewritten <draft_questions> in Bengali."""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_rewrite_prompt(draft_questions: str, problems: str, persona: dict) -> list[dict]:
    """
    Build a rewrite prompt when the draft questions fail the CoT reflection.

    Args:
        draft_questions: The flagged draft questions.
        problems: Description of the problems found.
        persona: Dict with persona details.

    Returns:
        OpenAI-compatible messages list for the rewrite call.
    """
    if "json_metadata" in persona and isinstance(persona["json_metadata"], dict):
        metadata = persona["json_metadata"]
    elif "json_metadata" in persona and isinstance(persona["json_metadata"], str):
        metadata = json.loads(persona["json_metadata"])
    else:
        metadata = persona

    profession = metadata.get("profession", persona.get("profession", "unknown"))
    location = metadata.get("location", persona.get("location", "unknown"))
    education = metadata.get("education", "unknown")

    user_content = REWRITE_PROMPT_TEMPLATE.format(
        problems=problems,
        draft=draft_questions,
        profession=profession,
        location=location,
        education=education,
    )

    return [
        {"role": "system", "content": "You rewrite AI-sounding text to sound like a real human typed it on their phone in Bengali (বাংলা). Output ONLY the rewritten questions in Bengali script in <draft_questions> tags. No explanations. Some English words like NID, online, SMS are OK to mix in."},
        {"role": "user", "content": user_content},
    ]


def validate_and_refine(llm_response: str, persona: dict = None) -> tuple[str, str]:
    """
    Parse the LLM response, validate via reflection, and return final output.

    Workflow:
    1. Extract <draft_questions> and <reflection> from the response.
    2. Run an additional banned-word check on the draft questions.
    3. If the reflection verdict is FAIL or banned words are found,
       flag for rewrite (the caller handles the actual LLM re-call).
    4. Return the final question text and full CoT log.

    Args:
        llm_response: The full text response from the LLM.
        persona: Optional persona dict for context in the CoT log.

    Returns:
        Tuple of (final_question_text, cot_log).
        If the reflection fails, the cot_log will contain "NEEDS_REWRITE"
        as a signal to the caller.
    """
    cot_log_parts = []
    cot_log_parts.append("=== RAW LLM RESPONSE ===")
    cot_log_parts.append(llm_response)
    cot_log_parts.append("")

    # Extract the draft questions (use the LAST <draft_questions> block,
    # since a FAIL reflection may produce a rewrite)
    all_drafts = re.findall(r"<draft_questions>(.*?)</draft_questions>",
                            llm_response, re.DOTALL)

    if not all_drafts:
        logger.warning("No <draft_questions> tags found in LLM response.")
        cot_log_parts.append("WARNING: No <draft_questions> tags found.")
        # Fall back to using the entire response as the question
        question_text = llm_response.strip()
        cot_log = "\n".join(cot_log_parts)
        return question_text, cot_log

    # Use the last draft (post-rewrite if applicable)
    draft_questions = all_drafts[-1].strip()

    # Extract reflection
    reflection = _extract_tag(llm_response, "reflection")
    if reflection:
        cot_log_parts.append("=== REFLECTION ===")
        cot_log_parts.append(reflection)
        cot_log_parts.append("")

    # Check for banned words in the draft
    banned_found = _contains_banned_words(draft_questions)
    if banned_found:
        cot_log_parts.append(f"=== BANNED WORDS DETECTED: {', '.join(banned_found)} ===")

    # Determine if the output passes
    needs_rewrite = False
    rewrite_reasons = []

    # Check reflection verdict
    if reflection:
        verdict_match = re.search(r"Verdict:\s*(PASS|FAIL)", reflection, re.IGNORECASE)
        if verdict_match and verdict_match.group(1).upper() == "FAIL":
            needs_rewrite = True
            reason_match = re.search(r"Reason:\s*(.*?)$", reflection, re.MULTILINE)
            if reason_match:
                rewrite_reasons.append(f"Reflection: {reason_match.group(1).strip()}")

    # Check for banned words
    if banned_found:
        needs_rewrite = True
        rewrite_reasons.append(f"Banned words found: {', '.join(banned_found)}")

    if needs_rewrite:
        cot_log_parts.append("=== VERDICT: NEEDS_REWRITE ===")
        cot_log_parts.append("Reasons: " + "; ".join(rewrite_reasons))
        cot_log = "\n".join(cot_log_parts)
        # Return the draft but signal the caller to rewrite
        return draft_questions, "NEEDS_REWRITE|" + cot_log
    else:
        cot_log_parts.append("=== VERDICT: PASS ===")
        cot_log = "\n".join(cot_log_parts)
        return draft_questions, cot_log
