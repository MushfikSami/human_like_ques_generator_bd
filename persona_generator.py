"""
persona_generator.py — Matrix-Driven Persona Creation

Generates diverse Bangladeshi personas across demographic dimensions: region,
profession, age, gender, social status, education level, and government-service
pain points.

Coverage strategy:
  * `profession`, `pain_point`, and `education` are placed in a deterministic
    co-prime striding matrix (see _stratified_cells) so every value on each of
    those three axes is evenly covered.
  * `location` is sampled *probabilistically* from REGION_WEIGHTS so the national
    distribution stays realistic (Dhaka Metro dominant) while still surfacing
    hyper-local/marginalized environments (slums, camps, char areas).
  * `age`, `gender`, `social_status` are weighted-sampled per persona.

Each persona is assigned a unique random_seed for reproducibility — the same
seed always produces the same persona.
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

# Regions are sampled probabilistically (see REGION_WEIGHTS) rather than strided,
# so the national distribution stays realistic (Dhaka Metro dominant) while still
# surfacing hyper-local / marginalized environments. REGIONS and REGION_WEIGHTS
# MUST stay the same length and order. Weights are relative — rng.choices
# normalizes them — so they need not sum to any particular total.
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
    # Hyper-local / marginalized environments
    "Korail Slum (Dhaka)", "Bhashantek Bosti (Dhaka)", "Geneva Camp (Mohammadpur)",
    "Railway Colony (Chattogram)", "Sitakunda Shipbreaking Yard Area",
    "Char areas of Kurigram",
]

REGION_WEIGHTS = [
    # Divisions / major cities — Dhaka Metro dominant (~25% of total weight)
    25.0, 10.0, 5.0, 5.0, 4.0,
    4.0, 3.5, 3.5,
    # Districts & notable areas — a couple of larger metro-adjacent ones higher,
    # ordinary districts ~1 each
    2.5, 3.0, 3.0, 2.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
    # Special / rural areas
    1.0, 1.0, 1.5, 1.5, 1.5,
    # Hyper-local / marginalized environments — small but non-trivial shares
    3.0, 2.0, 1.5, 1.5, 2.0,
    2.0,
]

assert len(REGIONS) == len(REGION_WEIGHTS), (
    f"REGIONS ({len(REGIONS)}) and REGION_WEIGHTS ({len(REGION_WEIGHTS)}) must match"
)

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
    # Gig economy / micro-professions
    "Foodpanda/Pathao food delivery rider", "F-commerce (Facebook Live) clothing seller",
    "mobile banking (bKash/Nagad) agent", "freelance graphic designer (Upwork/Fiverr)",
    "YouTube/TikTok regional content creator", "Hijra (transgender) community member",
    # Non-Resident Bangladeshis (NRBs) & migrants
    "construction worker in Dubai (UAE)", "palm oil plantation worker in Malaysia",
    "convenience store clerk in Saudi Arabia", "student studying in North America/UK",
    "returnee migrant worker",
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
    # Modern digital services
    "Universal Pension Scheme (Prottoy/Surokkha) enrollment",
    "e-TIN registration and zero-return submission",
    "BDRIS (Birth and Death Registration) server downtime",
    "BRTA smart driving license biometric delay",
    "Probashi Kallyan Bank loan application",
    "reporting bKash fraud / cybercrime to DB police",
    "dual citizenship certificate for e-Passport",
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


def _generate_single_persona(seed: int, forced_cell: tuple = None) -> dict:
    """
    Generate a single persona dict using the given random seed.

    The seed is applied locally so that the same seed always produces
    the same persona, enabling full reproducibility.

    Args:
        seed: Unique random seed for this persona.
        forced_cell: Optional (profession, pain_point, education) triple assigned
            by the stratified coverage planner. When provided, those three
            dimensions are fixed; when None, they are weighted-sampled from the
            seed (legacy path). `location` is always sampled probabilistically
            from REGION_WEIGHTS regardless of this argument.

    Returns:
        Dict with all persona fields ready for DB insertion.
    """
    rng = random.Random(seed)

    # Sample demographics
    age_range = rng.choices(AGE_RANGES, weights=AGE_WEIGHTS, k=1)[0]
    age = rng.randint(age_range[0], age_range[1])
    gender = rng.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
    social_status = rng.choices(SOCIAL_STATUSES, weights=SOCIAL_STATUS_WEIGHTS, k=1)[0]

    # Region is always weighted-probabilistic (not part of the striding matrix).
    location = rng.choices(REGIONS, weights=REGION_WEIGHTS, k=1)[0]

    if forced_cell is not None:
        profession, pain_point, education = forced_cell
    else:
        profession = rng.choice(PROFESSIONS)
        pain_point = rng.choice(PAIN_POINTS)
        education = rng.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS, k=1)[0]

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


def _stratified_cells(count: int, rng: random.Random):
    """
    Build a coverage-driven list of (profession, pain_point, education) cells.

    Rather than sampling each dimension independently (which leaves large parts
    of the matrix unvisited and over-samples common combinations by chance), we
    walk the PROFESSIONS × PAIN_POINTS × EDUCATION_LEVELS space with co-prime
    striding so the `count` personas spread evenly across all three dimensions.

    Region is intentionally NOT strided here — it is sampled probabilistically
    from REGION_WEIGHTS in _generate_single_persona to keep a realistic national
    distribution.

    The multipliers (7, 13, 11) must each stay coprime with their array length
    for even coverage; this holds for the current array sizes (62 professions,
    55 pain points, 9 education levels).

    Args:
        count: Number of cells to produce.
        rng: Seeded RNG for reproducible shuffling.

    Returns:
        List of (profession, pain_point, education) tuples of length `count`.
    """
    professions = PROFESSIONS[:]
    pains = PAIN_POINTS[:]
    educations = EDUCATION_LEVELS[:]
    rng.shuffle(professions)
    rng.shuffle(pains)
    rng.shuffle(educations)

    cells = []
    for i in range(count):
        profession = professions[(i * 7) % len(professions)]
        pain = pains[(i * 13) % len(pains)]
        education = educations[(i * 11) % len(educations)]
        cells.append((profession, pain, education))
    return cells


def generate_personas(count: int = 25000):
    """
    Generate `count` diverse Bangladeshi personas and bulk-insert them.

    Coverage-driven: the profession / pain-point / education triples are laid
    out with co-prime striding (see _stratified_cells) so those axes are evenly
    covered. Region is weighted-probabilistic; age, gender, social status and
    backstory are seeded per-persona for reproducibility.

    Args:
        count: Number of personas to generate (default 25,000).
    """
    logger.info("Generating %d personas (stratified coverage)...", count)
    db.init_pool()
    conn = db.get_connection()

    seed_rng = random.Random(42)  # Master seed for reproducibility
    seeds = [seed_rng.randint(0, 2**31) for _ in range(count)]
    cells = _stratified_cells(count, random.Random(43))

    try:
        buffer = []
        for i, (seed, cell) in enumerate(zip(seeds, cells)):
            persona = _generate_single_persona(seed, forced_cell=cell)
            buffer.append(persona)

            if len(buffer) >= 500:
                db.bulk_insert_personas(conn, buffer)
                logger.info("Inserted %d / %d personas", i + 1, count)
                buffer = []

        if buffer:
            db.bulk_insert_personas(conn, buffer)

        logger.info("Successfully generated and inserted %d personas.", count)
    except Exception:
        logger.exception("Error during persona generation.")
        raise
    finally:
        db.put_connection(conn)
