"""
persona_generator.py — Matrix-Driven Persona Creation

Generates diverse Bangladeshi personas using weighted random sampling across
demographic dimensions: region, profession, age, gender, social status,
education level, and government-service pain points.

Each persona is assigned a unique random_seed for reproducibility — the same
seed should produce the same LLM output when used during question generation.
"""

import json
import random
import logging

import db

logger = logging.getLogger(__name__)

# ─── Demographic Arrays ─────────────────────────────────────────────────────
# These arrays are designed to provide comprehensive coverage of Bangladesh's
# demographic landscape, weighted towards populations most likely to interact
# with government services.

REGIONS = [
    # Divisions / major cities
    "Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet",
    "Rangpur", "Barishal", "Mymensingh",
    # Districts & notable areas
    "Comilla", "Gazipur", "Narayanganj", "Cox's Bazar", "Bogra",
    "Jessore", "Dinajpur", "Tangail", "Narsingdi", "Faridpur",
    "Kurigram", "Bhola", "Patuakhali", "Noakhali", "Brahmanbaria",
    "Kishoreganj", "Habiganj", "Moulvibazar", "Sunamganj", "Netrokona",
    "Sherpur", "Jamalpur", "Chapainawabganj", "Naogaon", "Natore",
    "Pabna", "Sirajganj", "Joypurhat", "Thakurgaon", "Panchagarh",
    "Lalmonirhat", "Nilphamari", "Gaibandha", "Kushtia", "Meherpur",
    "Chuadanga", "Jhenaidah", "Magura", "Narail", "Satkhira",
    "Bagerhat", "Pirojpur", "Barguna", "Jhalokathi", "Chandpur",
    "Lakshmipur", "Feni", "Khagrachhari", "Rangamati", "Bandarban",
    "Manikganj", "Munshiganj", "Shariatpur", "Madaripur", "Gopalganj",
    # Special / rural areas
    "Chittagong Hill Tracts", "Sundarbans adjacent area", "Char areas (Jamuna)",
    "Haor region (Sunamganj)", "Coastal belt (Bhola)",
]

PROFESSIONS = [
    # Informal / rural
    "RMG worker", "rickshaw puller", "day labourer", "shrimp farmer",
    "rice farmer", "fisherman", "tea garden worker", "brick kiln worker",
    "street food vendor", "small shopkeeper", "vegetable seller",
    "auto-rickshaw driver", "boat operator", "tailor", "carpenter",
    "bamboo craftsman", "pottery maker", "seasonal migrant worker",
    # Semi-formal / trades
    "schoolteacher", "madrasa teacher", "private tutor", "nurse",
    "pharmacy assistant", "electrician", "plumber", "mobile repair technician",
    "beauty parlour owner", "garments supervisor",
    # Formal / professional
    "government clerk", "bank officer", "police constable",
    "university student", "college lecturer", "lawyer", "doctor",
    "engineer", "journalist", "NGO field worker",
    # Tech / modern
    "tech startup employee", "freelance web developer", "Uber/Pathao driver",
    "e-commerce seller", "social media manager", "call centre agent",
    # Specific demographics
    "expatriate worker's wife", "retired army personnel", "widow on pension",
    "disabled person (on social safety net)", "domestic worker",
    "madrasa student", "unemployed youth",
]

PAIN_POINTS = [
    # Identity & registration
    "NID correction", "birth certificate", "death certificate",
    "voter ID issue", "passport renewal", "passport application delay",
    "marriage certificate", "citizenship certificate",
    # Land & property
    "land registration", "land dispute resolution", "mutation of land records",
    "khas land allocation", "eviction notice",
    # Social safety nets
    "education stipend", "widow allowance", "old age allowance",
    "disability allowance", "freedom fighter allowance",
    "VGD/VGF card issue", "social safety net enrollment",
    # Agriculture & rural
    "agricultural subsidies", "fertilizer card", "crop damage compensation",
    "fisheries license", "livestock vaccination",
    # Utilities & services
    "electricity billing dispute", "gas connection", "water supply complaint",
    "mobile court complaint", "municipality tax",
    # Education
    "school admission", "SSC/HSC result correction", "scholarship application",
    "student loan", "university admission",
    # Legal & justice
    "police report (GD)", "court case status", "bail information",
    "legal aid application", "dowry complaint",
    # Health
    "hospital referral", "free medicine programme", "vaccination schedule",
    "disability certification",
    # Migration & foreign
    "BMET registration", "foreign employment visa", "remittance issue",
    "embassy appointment",
]

GENDERS = ["male", "female", "other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]

SOCIAL_STATUSES = [
    "lower-income", "lower-middle-income", "middle-income",
    "upper-middle-income", "upper-income",
]
SOCIAL_STATUS_WEIGHTS = [0.30, 0.30, 0.25, 0.10, 0.05]

EDUCATION_LEVELS = [
    "no formal education", "primary (class 1-5)", "secondary (class 6-10)",
    "SSC pass", "HSC pass", "bachelor's degree", "master's degree",
    "technical/vocational", "madrasa education",
]
EDUCATION_WEIGHTS = [0.15, 0.20, 0.20, 0.15, 0.12, 0.08, 0.03, 0.04, 0.03]

# Age distribution skewed towards younger population (Bangladesh demographics)
AGE_RANGES = [
    (18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 80),
]
AGE_WEIGHTS = [0.25, 0.30, 0.20, 0.12, 0.08, 0.05]

# ─── Backstory Templates ────────────────────────────────────────────────────
# These templates are filled with persona details to create a grounding
# narrative for the LLM prompt.

BACKSTORY_TEMPLATES = [
    "A {age}-year-old {gender} {profession} from {location}. {education_detail} "
    "Currently dealing with {pain_point}. {life_detail}",

    "{profession} based in {location}, age {age}. {education_detail} "
    "Has been struggling with {pain_point} for a while. {life_detail}",

    "Lives in {location}, works as a {profession}. {age} years old, {gender}. "
    "{education_detail} Needs help with {pain_point}. {life_detail}",

    "A {social_status} {profession} from {location}. {gender}, aged {age}. "
    "{education_detail} Frustrated about {pain_point}. {life_detail}",
]

LIFE_DETAILS = [
    "Has a family of {family_size} to support.",
    "Recently moved to {location} for work.",
    "Has been in this profession for {years_exp} years.",
    "Sole earner in the family.",
    "Supports elderly parents back in the village.",
    "First in the family to use the internet.",
    "Learned about online government services from a neighbour.",
    "Previously tried to resolve this issue at the local union parishad office.",
    "Heard about this service on community radio.",
    "A friend told them about using apps for government services.",
    "Tired of going to the government office repeatedly.",
    "Has limited access to the internet — mostly uses mobile data.",
    "Cannot take time off work to visit the government office.",
    "Has been waiting for months with no response.",
    "Was sent from one office to another without resolution.",
]


def _weighted_choice(items, weights=None):
    """Select a single item using weighted random sampling."""
    if weights:
        return random.choices(items, weights=weights, k=1)[0]
    return random.choice(items)


def _generate_single_persona(seed: int) -> dict:
    """
    Generate a single persona dict using the given random seed.

    The seed is applied locally so that the same seed always produces
    the same persona, enabling full reproducibility.

    Args:
        seed: Unique random seed for this persona.

    Returns:
        Dict with all persona fields ready for DB insertion.
    """
    rng = random.Random(seed)

    # Sample demographics
    age_range = rng.choices(AGE_RANGES, weights=AGE_WEIGHTS, k=1)[0]
    age = rng.randint(age_range[0], age_range[1])
    gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
    location = rng.choice(REGIONS)
    profession = rng.choice(PROFESSIONS)
    social_status = rng.choices(SOCIAL_STATUSES, weights=SOCIAL_STATUS_WEIGHTS, k=1)[0]
    education = rng.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS, k=1)[0]
    pain_point = rng.choice(PAIN_POINTS)

    # Build backstory
    education_detail = f"Education: {education}."
    life_detail = rng.choice(LIFE_DETAILS).format(
        location=location,
        family_size=rng.randint(2, 8),
        years_exp=rng.randint(1, 30),
    )
    template = rng.choice(BACKSTORY_TEMPLATES)
    backstory = template.format(
        age=age,
        gender=gender,
        profession=profession,
        location=location,
        social_status=social_status,
        education_detail=education_detail,
        pain_point=pain_point,
        life_detail=life_detail,
    )

    # Build the full metadata JSON
    json_metadata = {
        "age": age,
        "gender": gender,
        "location": location,
        "profession": profession,
        "social_status": social_status,
        "education": education,
        "pain_point": pain_point,
        "random_seed": seed,
    }

    return {
        "age": age,
        "gender": gender,
        "location": location,
        "profession": profession,
        "social_status": social_status,
        "backstory": backstory,
        "json_metadata": json.dumps(json_metadata),
    }


def generate_personas(count: int = 25000):
    """
    Generate `count` diverse Bangladeshi personas and insert them into the DB.

    Uses unique random seeds to ensure each persona is reproducible.
    Personas are inserted one-by-one with transactional safety so the
    process can be interrupted and resumed.

    Args:
        count: Number of personas to generate (default 25,000).
    """
    logger.info("Generating %d personas...", count)
    conn = db.get_connection()

    # Generate unique seeds upfront
    seed_rng = random.Random(42)  # Master seed for reproducibility
    seeds = [seed_rng.randint(0, 2**31) for _ in range(count)]

    try:
        for i, seed in enumerate(seeds):
            persona = _generate_single_persona(seed)
            persona_id = db.insert_persona(conn, persona)

            if (i + 1) % 500 == 0:
                logger.info("Inserted %d / %d personas (latest persona_id=%d)",
                            i + 1, count, persona_id)

        logger.info("Successfully generated and inserted %d personas.", count)
    except Exception:
        logger.exception("Error during persona generation at index %d", i)
        raise
    finally:
        conn.close()
