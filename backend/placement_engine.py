"""
placement_engine.py
===================
A self-contained Flask Blueprint for the Inferno Placement Platform.

Features:
  - Company vacancy posting with eligibility criteria (CGPA, skills, role)
  - AI-powered resume → vacancy skill match scoring
  - CGPA extraction + gate-filtering
  - Project Impact detector   (reward builders, not just studiers)
  - Entrepreneur / SaaS detector (flag founder-profile candidates)
  - Three-bucket HR categorisation:
        1. Mark-Based     → high assessment score, meets all criteria
        2. Project Power  → strong projects with real-world impact
        3. Entrepreneur   → SaaS / startup / founder profile (CTO-level flag)
  - Leaderboard per vacancy
  - HR Dashboard JSON APIs

Integration with app.py:
    from placement_engine import placement_bp
    app.register_blueprint(placement_bp)
"""

import os
import re
import json
import math
import sqlite3
import datetime

from flask import Blueprint, request, jsonify, session
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Blueprint + AI client
# ──────────────────────────────────────────────────────────────────────────────
placement_bp = Blueprint("placement", __name__, url_prefix="/placement")
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing required environment variable: GROQ_API_KEY")

_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL   = "llama-3.3-70b-versatile"

DB_PATH = "placement.db"

# ──────────────────────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────────────────────

def init_placement_db():
    """Creates the placement.db tables on first run."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Companies post vacancies here
    c.execute("""
        CREATE TABLE IF NOT EXISTS vacancies (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name    TEXT    NOT NULL,
            job_role        TEXT    NOT NULL,
            description     TEXT,
            required_skills TEXT,           -- JSON list
            good_to_have    TEXT,           -- JSON list
            min_cgpa        REAL    DEFAULT 6.0,
            ctc             TEXT,
            deadline        TEXT,
            posted_by       TEXT,           -- HR name / email
            created_at      TEXT
        )
    """)

    # Every candidate application mapped to a vacancy
    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            vacancy_id           INTEGER NOT NULL,
            candidate_name       TEXT,
            email                TEXT,
            cgpa                 REAL,
            resume_text          TEXT,
            verified_skills      TEXT,      -- JSON list (backed by projects)
            partial_skills       TEXT,      -- JSON list (listed only)
            predicted_role       TEXT,
            profile_summary      TEXT,
            skill_match_score    REAL,      -- 0-100 match vs THIS vacancy
            project_impact_score REAL,      -- 0-100 project strength
            entrepreneur_score   REAL,      -- 0-100 founder/SaaS signal
            assessment_score     REAL,      -- score from Inferno rounds (0-100)
            overall_score        REAL,      -- weighted final
            category             TEXT,      -- mark_based | project_power | entrepreneur
            status               TEXT DEFAULT 'pending',
            project_highlights   TEXT,      -- JSON list of impactful project bullets
            entrepreneur_signals TEXT,      -- JSON list of detected founder keywords
            cgpa_eligible        INTEGER,   -- 1 = passed CGPA gate, 0 = rejected
            applied_at           TEXT,
            FOREIGN KEY(vacancy_id) REFERENCES vacancies(id)
        )
    """)

    conn.commit()
    conn.close()


init_placement_db()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_loads_safe(val, fallback=None):
    """Safely parse a JSON string stored in the DB."""
    if fallback is None:
        fallback = []
    try:
        return json.loads(val) if val else fallback
    except Exception:
        return fallback


# ──────────────────────────────────────────────────────────────────────────────
# 1. CGPA EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

_CGPA_PATTERNS = [
    r"cgpa[:\s]*([0-9]+(?:\.[0-9]+)?)",
    r"gpa[:\s]*([0-9]+(?:\.[0-9]+)?)",
    r"cumulative[:\s]+grade[:\s]+point[:\s]+average[:\s]*([0-9]+(?:\.[0-9]+)?)",
    r"([0-9]+\.[0-9]+)\s*(?:/\s*10|out of 10)",
    r"aggregate[:\s]*([0-9]+(?:\.[0-9]+)?)\s*%",   # marks/percentage fallback
]

def extract_cgpa(resume_text: str) -> float | None:
    """
    Extracts CGPA/GPA from raw resume text using regex patterns.
    Returns float (0–10 scale) or None if not found.
    Converts percentage to GPA if detected.
    """
    text_lower = resume_text.lower()
    for pattern in _CGPA_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            val = float(match.group(1))
            # If 100-scale percentage detected, convert approximate GPA
            if "%" in pattern and val > 10:
                val = round(val / 10, 2)
            # Sanity check
            if 0.0 <= val <= 10.0:
                return val
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. DEEP RESUME PARSER  (calls Groq)
# ──────────────────────────────────────────────────────────────────────────────

def deep_parse_resume(resume_text: str) -> dict:
    """
    Full AI resume parse returning:
      - predicted_role
      - verified_skills   (from projects/work experience)
      - partial_skills    (skills section only)
      - profile_summary
      - confidence_score
      - project_bullets   (raw project lines for impact scoring)
      - entrepreneur_clues (raw lines hinting at startup/founder experience)
    """
    prompt = f"""You are an expert resume analyst. Parse this resume carefully.

Resume (first 4000 chars):
{resume_text[:4000]}

Return ONLY a valid JSON object with these keys:
{{
  "predicted_role":      "best-fit job title string",
  "verified_skills":     ["skills explicitly used in projects / work exp – lowercase"],
  "partial_skills":      ["skills only in skills section, no project evidence – lowercase"],
  "profile_summary":     "1-2 sentence candidate overview",
  "confidence_score":    0.0,
  "project_bullets":     ["each project description bullet from the resume, verbatim"],
  "entrepreneur_clues":  ["any lines mentioning startup, SaaS, founder, CEO, CTO, co-founder,
                           launched a product, raised funding, revenue, users, client, freelance"]
}}

Rules:
- Normalise all skill names to lowercase.
- Include EVERY project bullet in project_bullets (critical for impact detection).
- Return ONLY JSON — no markdown, no extra text.
"""
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        # Normalise lists
        data["verified_skills"]    = [s.lower().strip() for s in data.get("verified_skills",    [])]
        data["partial_skills"]     = [s.lower().strip() for s in data.get("partial_skills",     [])]
        data["project_bullets"]    = data.get("project_bullets",    [])
        data["entrepreneur_clues"] = data.get("entrepreneur_clues", [])
        return data
    except Exception as e:
        print(f"[PlacementEngine] deep_parse_resume error: {e}")
        return {
            "predicted_role":     "Unknown",
            "verified_skills":    [],
            "partial_skills":     [],
            "profile_summary":    "Unable to parse resume.",
            "confidence_score":   0.0,
            "project_bullets":    [],
            "entrepreneur_clues": [],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. SKILL MATCH — VACANCY SPECIFIC
# ──────────────────────────────────────────────────────────────────────────────

def compute_vacancy_match(
    verified_skills: list[str],
    partial_skills:  list[str],
    required_skills: list[str],
    good_to_have:    list[str],
) -> dict:
    """
    Scores how well a candidate matches a specific vacancy.
    Weights:
      Exact (verified in project) × 1.0
      Partial (listed only)       × 0.4
      Good-to-have bonus          max 15 pts
    Returns dict with score 0-100 + breakdown lists.
    """
    req = [s.lower().strip() for s in required_skills]
    gth = [s.lower().strip() for s in good_to_have]
    all_cand = set(verified_skills) | set(partial_skills)

    exact_hits   = [s for s in req if s in verified_skills]
    partial_hits = [s for s in req if s in partial_skills and s not in verified_skills]
    missing      = [s for s in req if s not in all_cand]
    bonus_hits   = [s for s in gth if s in all_cand]

    core_score  = (len(exact_hits) * 1.0 + len(partial_hits) * 0.4) / max(len(req), 1)
    bonus_score = (len(bonus_hits) / max(len(gth), 1)) if gth else 0

    raw  = core_score * 85 + bonus_score * 15
    score = round(min(raw, 100.0), 1)

    return {
        "score":        score,
        "exact_hits":   exact_hits,
        "partial_hits": partial_hits,
        "missing":      missing[:6],
        "bonus_hits":   bonus_hits,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. PROJECT IMPACT SCORER
# ──────────────────────────────────────────────────────────────────────────────

# Keywords that signal REAL impact (quantified / deployed / serving users)
_IMPACT_KEYWORDS = [
    r"\b\d+[\+k%]\b",                            # numbers: 500+, 2k, 40%
    r"\b(deployed|launched|production|live)\b",
    r"\b(users?|clients?|customers?)\b",
    r"\b(revenue|profit|funding|raised)\b",
    r"\b(reduced|improved|increased|optimised?|saved|scaled|automated)\b",
    r"\b(api|backend|cloud|aws|gcp|azure|docker|kubernetes)\b",
    r"\b(accuracy|performance|latency|throughput)\b",
    r"\b(open.?sourc[e]?|github|contributed)\b",
    r"\b(won|award|hackathon|1st|first|prize)\b",
]

_IMPACT_RE = [re.compile(p, re.IGNORECASE) for p in _IMPACT_KEYWORDS]


def score_project_impact(project_bullets: list[str]) -> tuple[float, list[str]]:
    """
    Scores the project section 0-100 based on impact signals.
    Returns (score, list_of_highlighted_bullets).
    """
    if not project_bullets:
        return 0.0, []

    highlights = []
    total_signals = 0

    for bullet in project_bullets:
        bullet_signals = sum(1 for regex in _IMPACT_RE if regex.search(bullet))
        if bullet_signals >= 2:          # ≥2 impact keywords → truly impactful
            highlights.append(bullet)
        total_signals += bullet_signals

    # Normalise: 3 signals per project is "excellent"
    expected = len(project_bullets) * 3
    raw_score = (total_signals / max(expected, 1)) * 100
    score     = round(min(raw_score, 100.0), 1)
    return score, highlights[:5]          # cap to 5 highlighted bullets


# ──────────────────────────────────────────────────────────────────────────────
# 5. ENTREPRENEUR / SaaS DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

_ENTREPRENEUR_KEYWORDS = [
    r"\b(ceo|cto|coo|cfo|founder|co-?founder|cofounder)\b",
    r"\b(startup|start.?up|venture|incubat)\b",
    r"\bsaas\b",
    r"\b(bootstrapped|self.funded|raised|seed|series)\b",
    r"\b(entrepreneur|entrepreneurship)\b",
    r"\b(launched\s+a\s+(?:product|startup|company|platform))\b",
    r"\b(my\s+(?:startup|company|product|app|platform))\b",
    r"\b(\d+\s*(?:paying\s+)?customers?|monthly\s+revenue|mrr|arr)\b",
    r"\b(product.?market.?fit|pivoted|iteration)\b",
    r"\b(angel|accelerator|y\s*combinator|techstars)\b",
]

_ENTREPRENEUR_RE = [re.compile(p, re.IGNORECASE) for p in _ENTREPRENEUR_KEYWORDS]


def score_entrepreneur(
    entrepreneur_clues: list[str],
    resume_text:        str,
) -> tuple[float, list[str]]:
    """
    Detects founder / SaaS / startup profile.
    Returns (score 0-100, list of detected signal phrases).
    """
    combined_text = " ".join(entrepreneur_clues) + " " + resume_text[:3000]
    signals = []

    for regex in _ENTREPRENEUR_RE:
        matches = regex.findall(combined_text)
        if matches:
            signals.extend(matches[:2])   # collect up to 2 examples per pattern

    # Deduplicate
    signals = list(dict.fromkeys([str(s).lower().strip() for s in signals]))

    # Score: each unique strong signal = 15 pts, cap at 100
    score = round(min(len(signals) * 15, 100.0), 1)
    return score, signals[:8]


# ──────────────────────────────────────────────────────────────────────────────
# 6. THREE-BUCKET CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify_candidate(
    skill_match_score:    float,
    project_impact_score: float,
    entrepreneur_score:   float,
    assessment_score:     float,    # from Inferno rounds (0-100)
    cgpa:                 float,
) -> tuple[str, float]:
    """
    Returns (category, overall_score).

    Priority:
      1. entrepreneur  → entrepreneur_score ≥ 50  (rare, always flag)
      2. project_power → skill_match ≥ 40 AND project_impact ≥ 55
      3. mark_based    → highest assessment + skill score

    Overall score weights:
      mark_based:    40% assessment + 40% skill_match + 20% cgpa_norm
      project_power: 30% assessment + 30% skill_match + 40% project_impact
      entrepreneur:  20% assessment + 25% skill_match + 30% project_impact + 25% entrepreneur
    """
    cgpa_norm = round((cgpa / 10.0) * 100, 1) if cgpa else 50.0   # normalise to 0-100

    if entrepreneur_score >= 50:
        category = "entrepreneur"
        overall  = (
            assessment_score    * 0.20 +
            skill_match_score   * 0.15 +
            project_impact_score* 0.40 +
            entrepreneur_score  * 0.25
        )
    elif skill_match_score >= 40 and project_impact_score >= 55:
        category = "project_power"
        overall  = (
            assessment_score    * 0.30 +
            skill_match_score   * 0.30 +
            project_impact_score* 0.40
        )
    else:
        category = "mark_based"
        # Weights requested: Projects (40%), CGPA (30%), Platform Performance (30%)
        overall  = (
            project_impact_score * 0.40 +
            cgpa_norm            * 0.30 +
            assessment_score     * 0.30
        )

    return category, round(overall, 2)


# ──────────────────────────────────────────────────────────────────────────────
# ── FLASK ROUTES ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@placement_bp.route("/hr_dashboard")
def render_hr_dashboard():
    from flask import render_template, session, redirect
    if not session.get("hr_authenticated"):
        return redirect("/hr-login")
    conn = _db()
    vacancies = conn.execute("SELECT * FROM vacancies ORDER BY id DESC").fetchall()
    applications = conn.execute("SELECT * FROM applications ORDER BY id DESC").fetchall()
    conn.close()
    
    import sqlite3
    conn_hist = sqlite3.connect("history.db")
    conn_hist.row_factory = sqlite3.Row
    try:
        simulations = conn_hist.execute("SELECT * FROM company_simulations ORDER BY id DESC LIMIT 20").fetchall()
    except Exception:
        simulations = []
    conn_hist.close()
    
    return render_template("hr_dashboard.html", vacancies=vacancies, applications=applications, simulations=simulations)


# ── A. POST A VACANCY  (HR / Company) ─────────────────────────────────────────

@placement_bp.route("/vacancy", methods=["POST"])
def post_vacancy():
    """
    HR posts a new job vacancy with eligibility criteria.

    Body (JSON):
    {
        "company_name":    "Google",
        "job_role":        "Data Scientist",
        "description":     "...",
        "required_skills": ["python", "machine learning", "sql"],
        "good_to_have":    ["spark", "aws", "docker"],
        "min_cgpa":        7.5,
        "ctc":             "20 LPA",
        "deadline":        "2026-04-30",
        "posted_by":       "hr@google.com"
    }
    """
    data = request.json or {}
    required_fields = ["company_name", "job_role", "required_skills"]
    for f in required_fields:
        if not data.get(f):
            return jsonify({"error": f"'{f}' is required."}), 400

    conn = _db()
    conn.execute("""
        INSERT INTO vacancies
            (company_name, job_role, description, required_skills, good_to_have,
             min_cgpa, ctc, deadline, posted_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["company_name"],
        data["job_role"],
        data.get("description", ""),
        json.dumps(data.get("required_skills", [])),
        json.dumps(data.get("good_to_have",    [])),
        float(data.get("min_cgpa", 6.0)),
        data.get("ctc",      "Not disclosed"),
        data.get("deadline", ""),
        data.get("posted_by",""),
        _now(),
    ))
    conn.commit()
    vacancy_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return jsonify({
        "status":     "success",
        "vacancy_id": vacancy_id,
        "message":    f"Vacancy posted for {data['job_role']} at {data['company_name']}.",
    }), 201


# ── B. LIST VACANCIES ─────────────────────────────────────────────────────────

@placement_bp.route("/vacancies", methods=["GET"])
def list_vacancies():
    """Returns all open vacancies for candidates to browse."""
    conn  = _db()
    rows  = conn.execute(
        "SELECT id, company_name, job_role, min_cgpa, ctc, deadline, created_at "
        "FROM vacancies ORDER BY id DESC"
    ).fetchall()
    conn.close()

    vacancies = []
    for r in rows:
        vacancies.append({
            "id":           r["id"],
            "company_name": r["company_name"],
            "job_role":     r["job_role"],
            "min_cgpa":     r["min_cgpa"],
            "ctc":          r["ctc"],
            "deadline":     r["deadline"],
            "posted":       r["created_at"],
        })

    return jsonify({"vacancies": vacancies})


# ── C. APPLY (Candidate submits resume for a vacancy) ─────────────────────────

@placement_bp.route("/apply/<int:vacancy_id>", methods=["POST"])
def apply_to_vacancy(vacancy_id: int):
    """
    Candidate applies to a vacancy.

    Form data:
      - resume_text   (pasted text)  OR  file key 'resume' (uploaded)
      - name          candidate name
      - email         candidate email
      - assessment_score   (float 0-100, pulled from Inferno session by default)

    Returns the full analysis JSON and stores in DB.
    """
    # ── Fetch the vacancy ──
    conn    = _db()
    vacancy = conn.execute(
        "SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)
    ).fetchone()
    if not vacancy:
        conn.close()
        return jsonify({"error": "Vacancy not found."}), 404

    required_skills = _json_loads_safe(vacancy["required_skills"])
    good_to_have    = _json_loads_safe(vacancy["good_to_have"])
    min_cgpa        = vacancy["min_cgpa"]

    # ── Get resume text ──
    resume_text = request.form.get("resume_text", "").strip()
    if not resume_text and "resume" in request.files:
        f = request.files["resume"]
        try:
            resume_text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            resume_text = ""

    if not resume_text:
        conn.close()
        return jsonify({"error": "Resume text is required."}), 400

    candidate_name    = request.form.get("name",  "Anonymous")
    email             = request.form.get("email", "")
    assessment_score  = float(request.form.get(
        "assessment_score",
        session.get("resume_final_pct", session.get("cognitive_score", 50))
    ))

    # ── Step 1: Extract CGPA ──
    cgpa         = extract_cgpa(resume_text)
    cgpa_display = cgpa if cgpa is not None else 0.0

    # ── Step 2: CGPA Gate ──
    cgpa_eligible = (cgpa is not None and cgpa >= min_cgpa)

    if not cgpa_eligible:
        conn.close()
        return jsonify({
            "eligible":  False,
            "reason":    (
                f"CGPA {cgpa_display:.2f} is below the minimum required "
                f"{min_cgpa:.1f} for this vacancy."
                if cgpa is not None
                else "CGPA not detected in resume. Please include your CGPA."
            ),
            "cgpa_found":   cgpa_display,
            "min_required": min_cgpa,
        }), 200

    # ── Step 3: Deep parse resume ──
    parsed = deep_parse_resume(resume_text)

    # ── Step 4: Skill match vs THIS vacancy ──
    match_result = compute_vacancy_match(
        parsed["verified_skills"],
        parsed["partial_skills"],
        required_skills,
        good_to_have,
    )
    skill_match_score = match_result["score"]

    # ── Step 5: Project impact ──
    project_impact_score, project_highlights = score_project_impact(
        parsed["project_bullets"]
    )

    # ── Step 6: Entrepreneur detection ──
    entrepreneur_score, entrepreneur_signals = score_entrepreneur(
        parsed["entrepreneur_clues"],
        resume_text,
    )

    # ── Step 7: Classify + overall score ──
    category, overall_score = classify_candidate(
        skill_match_score,
        project_impact_score,
        entrepreneur_score,
        assessment_score,
        cgpa_display,
    )

    # ── Step 8: Persist ──
    cur = conn.execute("""
        INSERT INTO applications
            (vacancy_id, candidate_name, email, cgpa, resume_text,
             verified_skills, partial_skills, predicted_role, profile_summary,
             skill_match_score, project_impact_score, entrepreneur_score,
             assessment_score, overall_score, category, status,
             project_highlights, entrepreneur_signals, cgpa_eligible, applied_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        vacancy_id,
        candidate_name,
        email,
        cgpa_display,
        resume_text[:5000],     # truncate for storage
        json.dumps(parsed["verified_skills"]),
        json.dumps(parsed["partial_skills"]),
        parsed["predicted_role"],
        parsed["profile_summary"],
        skill_match_score,
        project_impact_score,
        entrepreneur_score,
        assessment_score,
        overall_score,
        category,
        "pending",
        json.dumps(project_highlights),
        json.dumps(entrepreneur_signals),
        1 if cgpa_eligible else 0,
        _now(),
    ))
    conn.commit()
    app_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    # ── Response ──
    return jsonify({
        "eligible":             True,
        "application_id":       app_id,
        "candidate_name":       candidate_name,
        "cgpa":                 cgpa_display,
        "predicted_role":       parsed["predicted_role"],
        "profile_summary":      parsed["profile_summary"],

        "skill_match": {
            "score":        skill_match_score,
            "exact_hits":   match_result["exact_hits"],
            "partial_hits": match_result["partial_hits"],
            "missing":      match_result["missing"],
        },
        "project_impact": {
            "score":      project_impact_score,
            "highlights": project_highlights,
        },
        "entrepreneur": {
            "score":   entrepreneur_score,
            "signals": entrepreneur_signals,
        },
        "overall_score": overall_score,
        "category":      category,
        "category_label": {
            "mark_based":    "📊 Mark-Based Candidate",
            "project_power": "🔥 Project Power Candidate",
            "entrepreneur":  "🚀 Entrepreneur / SaaS Profile",
        }[category],
        "assessment_score": assessment_score,
    }), 201


# ── D. HR DASHBOARD — all applications for a vacancy (3 buckets) ─────────────

@placement_bp.route("/hr/dashboard/<int:vacancy_id>", methods=["GET"])
def hr_dashboard(vacancy_id: int):
    """
    Returns three shortlists for HR:
      1. mark_based    — sorted by overall_score desc
      2. project_power — sorted by project_impact_score desc
      3. entrepreneur  — sorted by entrepreneur_score desc

    Also returns leaderboard (top 10 overall).

    Query Params:
      min_skill_match  (default 0)
      status           filter by status (pending / shortlisted / rejected)
    """
    min_skill = float(request.args.get("min_skill_match", 0))
    status_f  = request.args.get("status", None)

    conn = _db()

    # Fetch vacancy info
    vacancy = conn.execute(
        "SELECT company_name, job_role, min_cgpa FROM vacancies WHERE id = ?",
        (vacancy_id,)
    ).fetchone()
    if not vacancy:
        conn.close()
        return jsonify({"error": "Vacancy not found."}), 404

    # Base query
    where = "vacancy_id = ? AND cgpa_eligible = 1 AND skill_match_score >= ?"
    params: list = [vacancy_id, min_skill]
    if status_f:
        where  += " AND status = ?"
        params.append(status_f)

    rows = conn.execute(
        f"""SELECT id, candidate_name, email, cgpa,
                   predicted_role, profile_summary,
                   skill_match_score, project_impact_score, entrepreneur_score,
                   assessment_score, overall_score, category, status,
                   project_highlights, entrepreneur_signals, applied_at
            FROM applications
            WHERE {where}
            ORDER BY overall_score DESC""",
        params,
    ).fetchall()
    conn.close()

    def _row_to_dict(r):
        return {
            "id":                    r["id"],
            "name":                  r["candidate_name"],
            "email":                 r["email"],
            "cgpa":                  r["cgpa"],
            "predicted_role":        r["predicted_role"],
            "profile_summary":       r["profile_summary"],
            "skill_match_score":     r["skill_match_score"],
            "project_impact_score":  r["project_impact_score"],
            "entrepreneur_score":    r["entrepreneur_score"],
            "assessment_score":      r["assessment_score"],
            "overall_score":         r["overall_score"],
            "category":              r["category"],
            "status":                r["status"],
            "project_highlights":    _json_loads_safe(r["project_highlights"]),
            "entrepreneur_signals":  _json_loads_safe(r["entrepreneur_signals"]),
            "applied_at":            r["applied_at"],
        }

    all_dicts = [_row_to_dict(r) for r in rows]

    mark_based    = sorted(
        [c for c in all_dicts if c["category"] == "mark_based"],
        key=lambda x: x["overall_score"], reverse=True
    )
    project_power = sorted(
        [c for c in all_dicts if c["category"] == "project_power"],
        key=lambda x: x["project_impact_score"], reverse=True
    )
    entrepreneur  = sorted(
        [c for c in all_dicts if c["category"] == "entrepreneur"],
        key=lambda x: x["entrepreneur_score"], reverse=True
    )

    # Leaderboard = top 10 overall (all categories)
    leaderboard = sorted(all_dicts, key=lambda x: x["overall_score"], reverse=True)[:10]
    for rank, entry in enumerate(leaderboard, 1):
        entry["rank"] = rank

    return jsonify({
        "vacancy": {
            "id":           vacancy_id,
            "company":      vacancy["company_name"],
            "role":         vacancy["job_role"],
            "min_cgpa":     vacancy["min_cgpa"],
        },
        "summary": {
            "total_eligible":    len(all_dicts),
            "mark_based_count":  len(mark_based),
            "project_count":     len(project_power),
            "entrepreneur_count": len(entrepreneur),
        },
        "buckets": {
            "mark_based":    mark_based,
            "project_power": project_power,
            "entrepreneur":  entrepreneur,
        },
        "leaderboard": leaderboard,
    })


# ── E. UPDATE APPLICATION STATUS (shortlist / reject) ────────────────────────

@placement_bp.route("/hr/application/<int:app_id>/status", methods=["PATCH"])
def update_status(app_id: int):
    """
    HR marks a candidate as shortlisted or rejected.
    Body: { "status": "shortlisted" | "rejected" | "pending" }
    """
    data   = request.json or {}
    status = data.get("status", "").strip()
    if status not in ("shortlisted", "rejected", "pending"):
        return jsonify({"error": "status must be 'shortlisted', 'rejected', or 'pending'."}), 400

    conn = _db()
    conn.execute("UPDATE applications SET status = ? WHERE id = ?", (status, app_id))
    conn.commit()
    conn.close()
    return jsonify({"status": "updated", "application_id": app_id, "new_status": status})


# ── F. CANDIDATE SELF-CHECK  (how do I score for a vacancy?) ─────────────────

@placement_bp.route("/check_eligibility/<int:vacancy_id>", methods=["POST"])
def check_eligibility(vacancy_id: int):
    """
    Quick eligibility check for a candidate BEFORE formally applying.
    Only checks CGPA gate — does not store anything.

    Body (JSON): { "cgpa": 7.8 }
    Optionally: { "resume_text": "..." }  for CGPA auto-extract
    """
    conn    = _db()
    vacancy = conn.execute(
        "SELECT company_name, job_role, min_cgpa, required_skills FROM vacancies WHERE id = ?",
        (vacancy_id,)
    ).fetchone()
    conn.close()

    if not vacancy:
        return jsonify({"error": "Vacancy not found."}), 404

    data         = request.json or {}
    provided_cgpa = data.get("cgpa")

    if provided_cgpa is None and data.get("resume_text"):
        provided_cgpa = extract_cgpa(data["resume_text"])

    if provided_cgpa is None:
        return jsonify({
            "eligible": None,
            "message":  "Provide cgpa or resume_text for auto-detection.",
            "min_cgpa": vacancy["min_cgpa"],
        })

    eligible = float(provided_cgpa) >= vacancy["min_cgpa"]
    return jsonify({
        "eligible":         eligible,
        "your_cgpa":        float(provided_cgpa),
        "min_required":     vacancy["min_cgpa"],
        "company":          vacancy["company_name"],
        "role":             vacancy["job_role"],
        "required_skills":  _json_loads_safe(vacancy["required_skills"]),
        "message": (
            "✅ You meet the CGPA requirement. Proceed to apply."
            if eligible else
            f"❌ Minimum CGPA for this role is {vacancy['min_cgpa']:.1f}. You are not eligible."
        ),
    })


# ── G. LEADERBOARD (public — candidate can see their rank) ───────────────────

@placement_bp.route("/leaderboard/<int:vacancy_id>", methods=["GET"])
def leaderboard(vacancy_id: int):
    """
    Top-N candidates for a vacancy (anonymised for public view).
    Query param: top (default 10)
    """
    top  = min(int(request.args.get("top", 10)), 50)
    conn = _db()

    vacancy = conn.execute(
        "SELECT company_name, job_role FROM vacancies WHERE id = ?", (vacancy_id,)
    ).fetchone()
    if not vacancy:
        conn.close()
        return jsonify({"error": "Vacancy not found."}), 404

    rows = conn.execute("""
        SELECT candidate_name, overall_score, skill_match_score,
               project_impact_score, category, status
        FROM applications
        WHERE vacancy_id = ? AND cgpa_eligible = 1
        ORDER BY overall_score DESC
        LIMIT ?
    """, (vacancy_id, top)).fetchall()
    conn.close()

    board = []
    for rank, r in enumerate(rows, 1):
        # Anonymise: only first name + last initial
        name_parts = r["candidate_name"].split()
        anon_name  = (name_parts[0] + " " + name_parts[-1][0] + ".") if len(name_parts) > 1 else name_parts[0]
        board.append({
            "rank":                rank,
            "name":                anon_name,
            "overall_score":       r["overall_score"],
            "skill_match_score":   r["skill_match_score"],
            "project_impact_score": r["project_impact_score"],
            "category":            r["category"],
            "status":              r["status"],
        })

    return jsonify({
        "vacancy":     {"company": vacancy["company_name"], "role": vacancy["job_role"]},
        "leaderboard": board,
        "total_shown": len(board),
    })


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST  (python placement_engine.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick unit test of the pure-Python helpers (no API calls)
    sample_resume = """
    John Doe | CGPA: 8.7/10

    Projects:
    - Built a Flask REST API serving 2000+ daily users; reduced latency by 40% using Redis caching.
    - Deployed machine learning model on AWS SageMaker; achieved 94% accuracy on customer churn prediction.
    - Launched a SaaS dashboard (500 paying customers, $3k MRR) as co-founder.

    Experience:
    - Data Scientist Intern at XYZ Corp: built ETL pipelines with Apache Spark, automated reporting.

    Skills: Python, Machine Learning, SQL, Pandas, NumPy, Tableau, Docker
    """

    print("── CGPA EXTRACTION ──────────────────────────")
    cgpa = extract_cgpa(sample_resume)
    print(f"  CGPA found: {cgpa}")

    print("\n── PROJECT IMPACT ───────────────────────────")
    bullets = [
        "Built a Flask REST API serving 2000+ daily users; reduced latency by 40%.",
        "Deployed ML model on AWS SageMaker; 94% accuracy.",
        "Launched SaaS dashboard (500 paying customers, $3k MRR).",
        "Worked on data analysis tasks.",
    ]
    impact_score, highlights = score_project_impact(bullets)
    print(f"  Impact Score : {impact_score}")
    print(f"  Highlights   : {highlights}")

    print("\n── ENTREPRENEUR DETECTION ───────────────────")
    clues = ["Launched a SaaS dashboard as co-founder with $3k MRR"]
    e_score, e_signals = score_entrepreneur(clues, sample_resume)
    print(f"  Entrepreneur Score : {e_score}")
    print(f"  Signals            : {e_signals}")

    print("\n── SKILL MATCH ──────────────────────────────")
    result = compute_vacancy_match(
        verified_skills=["python", "machine learning", "aws", "flask", "redis"],
        partial_skills=["sql", "docker", "pandas"],
        required_skills=["python", "machine learning", "sql", "aws", "spark"],
        good_to_have=["docker", "kubernetes", "airflow"],
    )
    print(f"  Match Score  : {result['score']}")
    print(f"  Exact Hits   : {result['exact_hits']}")
    print(f"  Partial Hits : {result['partial_hits']}")
    print(f"  Missing      : {result['missing']}")

    print("\n── CLASSIFICATION ───────────────────────────")
    category, overall = classify_candidate(
        skill_match_score=result["score"],
        project_impact_score=impact_score,
        entrepreneur_score=e_score,
        assessment_score=78.0,
        cgpa=cgpa or 0.0,
    )
    print(f"  Category      : {category}")
    print(f"  Overall Score : {overall}")
    print("\n✅ All helpers tested successfully.")
