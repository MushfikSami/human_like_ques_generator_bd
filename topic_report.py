"""
topic_report.py — Visual distribution report of generated question topics.

Queries the generated questions, rolls the 55 `pain_point` topics up into ~10
categories, and writes a self-contained `topic_report.html` (no external
dependencies, no CDN) with:
  * a stat-tile row (hero numbers),
  * a ranked category bar chart,
  * the full 55 topics ranked, sectioned by category,
  * a plain table view (accessibility).

Colours follow the dataviz skill's validated reference palette (blue sequential
for magnitude, zero baseline). Light + dark mode via prefers-color-scheme.
"""

import html
import logging
import os

import db

logger = logging.getLogger(__name__)

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topic_report.html")

# ─── Topic → category mapping (mirrors PAIN_POINTS groups in persona_generator) ──
TOPIC_CATEGORY = {
    # Identity & registration
    "NID correction": "Identity & registration",
    "birth certificate": "Identity & registration",
    "death certificate": "Identity & registration",
    "voter ID issue": "Identity & registration",
    "passport renewal": "Identity & registration",
    "passport application delay": "Identity & registration",
    "marriage certificate": "Identity & registration",
    "citizenship certificate": "Identity & registration",
    # Land & property
    "land registration": "Land & property",
    "land dispute resolution": "Land & property",
    "mutation of land records": "Land & property",
    "khas land allocation": "Land & property",
    "eviction notice": "Land & property",
    # Social safety nets
    "education stipend": "Social safety nets",
    "widow allowance": "Social safety nets",
    "old age allowance": "Social safety nets",
    "disability allowance": "Social safety nets",
    "freedom fighter allowance": "Social safety nets",
    "VGD/VGF card issue": "Social safety nets",
    "social safety net enrollment": "Social safety nets",
    # Agriculture & rural
    "agricultural subsidies": "Agriculture & rural",
    "fertilizer card": "Agriculture & rural",
    "crop damage compensation": "Agriculture & rural",
    "fisheries license": "Agriculture & rural",
    "livestock vaccination": "Agriculture & rural",
    # Utilities & services
    "electricity billing dispute": "Utilities & services",
    "gas connection": "Utilities & services",
    "water supply complaint": "Utilities & services",
    "mobile court complaint": "Utilities & services",
    "municipality tax": "Utilities & services",
    # Education
    "school admission": "Education",
    "SSC/HSC result correction": "Education",
    "scholarship application": "Education",
    "student loan": "Education",
    "university admission": "Education",
    # Legal & justice
    "police report (GD)": "Legal & justice",
    "court case status": "Legal & justice",
    "bail information": "Legal & justice",
    "legal aid application": "Legal & justice",
    "dowry complaint": "Legal & justice",
    # Health
    "hospital referral": "Health",
    "free medicine programme": "Health",
    "vaccination schedule": "Health",
    "disability certification": "Health",
    # Migration & foreign
    "BMET registration": "Migration & foreign",
    "foreign employment visa": "Migration & foreign",
    "remittance issue": "Migration & foreign",
    "embassy appointment": "Migration & foreign",
    # Modern digital services
    "Universal Pension Scheme (Prottoy/Surokkha) enrollment": "Modern digital services",
    "e-TIN registration and zero-return submission": "Modern digital services",
    "BDRIS (Birth and Death Registration) server downtime": "Modern digital services",
    "BRTA smart driving license biometric delay": "Modern digital services",
    "Probashi Kallyan Bank loan application": "Modern digital services",
    "reporting bKash fraud / cybercrime to DB police": "Modern digital services",
    "dual citizenship certificate for e-Passport": "Modern digital services",
}


# ─── Data ────────────────────────────────────────────────────────────────────

def _fetch_topic_counts(conn) -> list[tuple[str, int]]:
    """Return [(topic, count)] over generated questions, most frequent first."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT p.json_metadata->>'pain_point' AS topic, COUNT(*) AS n
            FROM {db.QUESTIONS_TABLE} q
            JOIN personas p USING (persona_id)
            GROUP BY topic
            ORDER BY n DESC;
        """)
        return [(t, n) for t, n in cur.fetchall()]


# ─── HTML rendering ──────────────────────────────────────────────────────────

_STYLE = """
:root{
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --baseline:#c3c2b7; --series:#2a78d6;
  --series-soft:#cde2fb; --border:rgba(11,11,11,.10);
}
@media (prefers-color-scheme:dark){:root{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --baseline:#383835; --series:#3987e5;
  --series-soft:#184f95; --border:rgba(255,255,255,.10);
}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}
.wrap{max-width:960px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:24px;margin:0 0 4px} .sub{color:var(--ink2);margin:0 0 28px;font-size:14px}
h2{font-size:16px;margin:40px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--grid)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:8px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.tile .v{font-size:26px;font-weight:650}
.tile .l{color:var(--ink2);font-size:12.5px;margin-top:2px}
.cat{margin:26px 0 6px;display:flex;justify-content:space-between;align-items:baseline}
.cat .n{font-weight:600;font-size:14.5px} .cat .s{color:var(--ink2);font-size:12.5px;font-variant-numeric:tabular-nums}
.row{display:grid;grid-template-columns:230px 1fr 96px;align-items:center;gap:12px;padding:3px 0}
.row .name{font-size:13px;color:var(--ink2);text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{position:relative;height:16px;background:var(--surface);border-radius:0 4px 4px 0}
.fill{height:100%;background:var(--series);border-radius:0 4px 4px 0;min-width:2px}
.row .val{font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
.catrow .name{font-weight:600;color:var(--ink)} .catrow .fill{background:var(--series)}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--ink2);font-weight:600} td.num{text-align:right;font-variant-numeric:tabular-nums}
details{margin-top:36px} summary{cursor:pointer;color:var(--ink2);font-size:14px}
.foot{margin-top:48px;color:var(--muted);font-size:12px}
"""


def _bar_row(name: str, count: int, scale_max: int, total: int, cls: str = "") -> str:
    pct = 100.0 * count / total if total else 0
    width = 100.0 * count / scale_max if scale_max else 0
    nm = html.escape(name)
    return (
        f'<div class="row {cls}" title="{nm}: {count:,} ({pct:.1f}%)">'
        f'<div class="name">{nm}</div>'
        f'<div class="bar"><div class="fill" style="width:{width:.2f}%"></div></div>'
        f'<div class="val">{count:,} · {pct:.1f}%</div></div>'
    )


def _render(topic_counts: list[tuple[str, int]], total_questions: int,
            personas: int) -> str:
    # Category aggregation
    cat_counts: dict[str, int] = {}
    cat_topics: dict[str, list[tuple[str, int]]] = {}
    for topic, n in topic_counts:
        cat = TOPIC_CATEGORY.get(topic, "Other")
        cat_counts[cat] = cat_counts.get(cat, 0) + n
        cat_topics.setdefault(cat, []).append((topic, n))

    cats_sorted = sorted(cat_counts.items(), key=lambda kv: kv[1], reverse=True)
    counts = [n for _, n in topic_counts]
    cmin, cmax = min(counts), max(counts)
    evenness = cmax / cmin if cmin else 0

    # Stat tiles
    tiles = [
        (f"{total_questions:,}", "Questions"),
        (f"{personas:,}", "Personas"),
        (f"{len(topic_counts)}", "Distinct topics"),
        (f"{len(cat_counts)}", "Categories"),
        (f"{cmin:,}–{cmax:,}", "Per-topic range"),
        (f"{evenness:.2f}×", "Max ÷ min (evenness)"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in tiles
    )

    # Category chart (scaled to largest category)
    cat_max = cats_sorted[0][1]
    cat_rows = "".join(
        _bar_row(name, n, cat_max, total_questions, cls="catrow")
        for name, n in cats_sorted
    )

    # Full 55, sectioned by category (each section's topics sorted desc,
    # bars scaled to the global max topic count so sections are comparable)
    sections = []
    for name, n in cats_sorted:
        share = 100.0 * n / total_questions
        topics = sorted(cat_topics[name], key=lambda kv: kv[1], reverse=True)
        rows = "".join(_bar_row(t, c, cmax, total_questions) for t, c in topics)
        sections.append(
            f'<div class="cat"><span class="n">{html.escape(name)}</span>'
            f'<span class="s">{n:,} · {share:.1f}% · {len(topics)} topics</span></div>{rows}'
        )
    sections_html = "".join(sections)

    # Table view
    trows = "".join(
        f'<tr><td>{html.escape(t)}</td><td>{html.escape(TOPIC_CATEGORY.get(t,"Other"))}</td>'
        f'<td class="num">{c:,}</td><td class="num">{100.0*c/total_questions:.2f}%</td></tr>'
        for t, c in topic_counts
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Question Topic Distribution</title>
<style>{_STYLE}</style></head>
<body><div class="wrap">
  <h1>Question Topic Distribution</h1>
  <p class="sub">Generated Bangladeshi government-service questions, grouped by topic (pain point).</p>
  <div class="tiles">{tiles_html}</div>

  <h2>By category</h2>
  {cat_rows}

  <h2>All topics, by category</h2>
  {sections_html}

  <details><summary>Table view (all {len(topic_counts)} topics)</summary>
    <table><thead><tr><th>Topic</th><th>Category</th><th>Count</th><th>Share</th></tr></thead>
    <tbody>{trows}</tbody></table>
  </details>

  <p class="foot">Bars scale to the largest value on each chart; a zero baseline is used throughout.
  The near-uniform per-topic counts reflect the stratified coverage design.</p>
</div></body></html>"""


# ─── Entry point ─────────────────────────────────────────────────────────────

def generate(output_path: str = OUTPUT_PATH) -> str:
    """Build the HTML report and write it to disk. Returns the output path."""
    db.init_pool()
    conn = db.get_connection()
    try:
        topic_counts = _fetch_topic_counts(conn)
        total = sum(n for _, n in topic_counts)
        personas = db.count_personas(conn, "done")
    finally:
        db.put_connection(conn)
        db.close_pool()

    if not topic_counts:
        logger.warning("No generated questions found — nothing to report.")
        return output_path

    n_cats = len({TOPIC_CATEGORY.get(t, "Other") for t, _ in topic_counts})
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_render(topic_counts, total, personas))

    print(f"Wrote {output_path}")
    print(f"  {total:,} questions across {len(topic_counts)} topics / {n_cats} categories")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate()
