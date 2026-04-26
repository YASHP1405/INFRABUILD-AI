"""
coding_verifier.py
==================
Flask Blueprint: Live Coding Verification Engine

The Judge's Idea:
  If a candidate claims they are proficient in programming but cannot handle a
  minor error during a live timed challenge, why should we recommend them to HR?

How it works:
  1. Candidate is given a coding task matched to their claimed skills
  2. Browser records: keystrokes/min, pause count, error count, time-to-solve, tab switches
  3. Candidate submits code + behavioral telemetry
  4. AI evaluates: code correctness + coding behavior pattern
  5. Follow-up: AI asks 3 questions about their own submitted code
  6. Final score = code quality (40%) + behavior (30%) + Q&A authenticity (30%)

Routes:
  POST /verify/start             → Generate a coding task
  POST /verify/submit            → Submit code + telemetry + get follow-up Qs
  POST /verify/answer_followup   → Answer the 3 follow-up questions → final score

Integration:
    from coding_verifier import verify_bp
    app.register_blueprint(verify_bp)
"""

import os
import re
import json
import sqlite3
import datetime

from flask import Blueprint, request, jsonify
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
verify_bp = Blueprint("verify", __name__, url_prefix="/verify")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing required environment variable: GROQ_API_KEY")

_ai   = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"
DB_PATH = "placement.db"

# ──────────────────────────────────────────────────────────────────────────────
# DB
# ──────────────────────────────────────────────────────────────────────────────

def _init_verify_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS coding_verifications (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_email      TEXT,
            claimed_skills       TEXT,
            task_title           TEXT,
            task_description     TEXT,
            time_limit_sec       INTEGER,
            submitted_code       TEXT,
            time_taken_sec       INTEGER,
            keystrokes_per_min   REAL,
            pause_count          INTEGER,
            error_events         INTEGER,
            tab_switches         INTEGER,
            code_score           REAL,
            behavior_score       REAL,
            followup_questions   TEXT,    -- JSON
            followup_answers     TEXT,    -- JSON
            followup_score       REAL,
            final_score          REAL,
            verdict              TEXT,
            created_at           TEXT
        )
    """)
    conn.commit()
    conn.close()

_init_verify_table()

# ──────────────────────────────────────────────────────────────────────────────
# TASK GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_coding_task(skills: list[str], difficulty: str = "medium") -> dict:
    """
    Generates a real coding task matched to the candidate's claimed skill set.
    Difficulty: easy | medium | hard
    """
    skill_str = ", ".join(skills[:5]) if skills else "Python"
    prompt = f"""You are designing a timed live coding challenge.

Candidate's claimed skills: {skill_str}
Difficulty: {difficulty}

Generate a single, focused coding task that:
1. Can be solved in 15-25 minutes by a genuinely proficient candidate.
2. Has a small, intentional bug or tricky edge case to see if they catch it.
3. Is specific to one of their claimed skills (not generic).
4. Has a clear expected output they can verify locally.

Return ONLY this JSON:
{{
  "title":       "Short task name",
  "description": "Full problem statement with input/output examples",
  "starter_code": "def solution(...):\\n    # Your code here\\n    pass",
  "expected_output": "What correct output looks like",
  "hint_trap": "The tricky part most candidates miss (DO NOT share with candidate)",
  "time_limit_sec": 1200
}}
No markdown. No extra text.
"""
    try:
        resp = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Verify] Task generation error: {e}")
        return {
            "title":        "Two Sum Variant",
            "description":  "Given a list of integers and a target, return all UNIQUE pairs that sum to target. Example: nums=[2,7,2,11], target=9 → [(2,7)]",
            "starter_code": "def find_pairs(nums, target):\n    # Your code here\n    pass",
            "expected_output": "List of unique tuples",
            "hint_trap":    "Duplicates in the list — most candidates miss deduplication",
            "time_limit_sec": 1200,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CODE EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_submitted_code(task: dict, code: str) -> tuple[float, str]:
    """
    AI evaluates the submitted code for correctness, edge-case handling,
    code style, and whether the candidate caught the 'hint_trap'.
    Returns (score 0-100, detailed_feedback).
    """
    prompt = f"""You are a senior code reviewer evaluating a live coding submission.

Task: {task['title']}
Description: {task['description']}
Expected output: {task['expected_output']}
Known tricky part: {task.get('hint_trap', 'N/A')}

Candidate's code:
```
{code[:2000]}
```

Evaluate on:
1. Correctness (0-40): Does it solve the problem correctly for all cases?
2. Edge cases (0-25): Does it handle empty input, duplicates, edge values?
3. Code quality (0-20): Is it readable, properly named, no redundant code?
4. Trap detection (0-15): Did they handle the known tricky part?

Return ONLY this JSON:
{{
  "correctness":    <0-40>,
  "edge_cases":     <0-25>,
  "code_quality":   <0-20>,
  "trap_detected":  <0-15>,
  "total_score":    <0-100>,
  "feedback":       "2-3 sentence critique of the code"
}}
"""
    try:
        resp = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        result = json.loads(raw)
        return float(result.get("total_score", 50)), result.get("feedback", "")
    except Exception as e:
        print(f"[Verify] Code eval error: {e}")
        return 50.0, "Automated evaluation temporarily unavailable."


# ──────────────────────────────────────────────────────────────────────────────
# BEHAVIOR SCORER
# ──────────────────────────────────────────────────────────────────────────────

def score_behavior(telemetry: dict, time_limit_sec: int) -> tuple[float, dict]:
    """
    Scores coding behavior from browser-collected telemetry.

    Telemetry fields:
      time_taken_sec    - actual time to submit
      keystrokes_per_min - typing fluency
      pause_count       - long pauses (>30s) = possible googling
      error_events      - syntax errors caught by editor
      tab_switches      - switching away (proctoring)

    Returns (behavior_score 0-100, breakdown_dict)
    """
    time_taken   = telemetry.get("time_taken_sec",    time_limit_sec)
    kpm          = telemetry.get("keystrokes_per_min", 30)
    pauses       = telemetry.get("pause_count",        0)
    errors       = telemetry.get("error_events",       0)
    tab_sw       = telemetry.get("tab_switches",       0)

    # ── Speed score (0-30): faster within time = better ──
    time_ratio   = min(time_taken / max(time_limit_sec, 1), 1.0)
    speed_score  = round((1 - time_ratio * 0.7) * 30, 1)   # max 30 pts

    # ── Fluency score (0-25): KPM between 20-80 is natural coding ──
    if   20 <= kpm <= 80:  fluency_score = 25
    elif 10 <= kpm < 20:   fluency_score = 15
    elif kpm > 80:         fluency_score = 18    # very fast, possible paste
    else:                  fluency_score = 5

    # ── Focus score (0-25): penalise tab switches + long pauses ──
    focus_score  = max(0, 25 - (tab_sw * 5) - (pauses * 2))
    focus_score  = round(min(focus_score, 25), 1)

    # ── Error recovery (0-20): some errors are GOOD (means actually typing) ──
    if   0  <= errors <= 3:  error_score = 20  # clean or minor errors
    elif 4  <= errors <= 8:  error_score = 12  # moderate — acceptable
    elif 9  <= errors <= 15: error_score = 6   # too many
    else:                    error_score = 2   # excessive

    behavior_score = round(speed_score + fluency_score + focus_score + error_score, 1)
    behavior_score = min(behavior_score, 100.0)

    breakdown = {
        "speed_score":   speed_score,
        "fluency_score": fluency_score,
        "focus_score":   focus_score,
        "error_score":   error_score,
        "total":         behavior_score,
        "flags": {
            "tab_switching": tab_sw > 2,
            "suspiciously_fast": kpm > 100,
            "too_many_pauses": pauses > 5,
            "excessive_errors": errors > 15,
        },
    }
    return behavior_score, breakdown


# ──────────────────────────────────────────────────────────────────────────────
# FOLLOW-UP QUESTION GENERATOR (from their OWN code)
# ──────────────────────────────────────────────────────────────────────────────

def generate_followup_questions(task: dict, code: str) -> list[str]:
    """
    Generates 3 follow-up questions based specifically on the candidate's
    submitted code — impossible to answer without having written it.
    """
    prompt = f"""A candidate submitted this code for the task: "{task['title']}"

Their code:
```
{code[:1500]}
```

Generate EXACTLY 3 follow-up interview questions about THIS specific code.
Rules:
1. Question 1: About a specific line or function they wrote — "Why did you use X here?"
2. Question 2: "What happens if the input is [specific edge case]? Walk me through your code."
3. Question 3: "How would you optimize this for 1 million inputs?"

These questions should be IMPOSSIBLE to answer without having written this exact code.
Return ONLY a JSON array: ["question1", "question2", "question3"]
"""
    try:
        resp = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)[:3]
    except Exception:
        return [
            "Walk me through the logic of your main function line by line.",
            "What happens if you pass an empty list to your solution?",
            "How would you refactor this to handle 10× the input size efficiently?",
        ]


def evaluate_followup_answers(
    code: str, questions: list[str], answers: list[str]
) -> tuple[float, list[str]]:
    """Scores follow-up answers for depth and code-specific accuracy."""
    qa = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    prompt = f"""Evaluate these 3 coding follow-up answers.
The candidate wrote this code:
```
{code[:1000]}
```

Q&A:
{qa}

Score each answer 0-10 for: does it accurately describe THEIR code? Is it specific?
Return ONLY a JSON array of 3 objects:
[{{"score": 8, "feedback": "Good — they correctly described the set usage"}}]
"""
    try:
        resp = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        results = json.loads(raw)
        avg     = sum(r.get("score", 5) for r in results) / max(len(results), 1)
        score   = round(avg * 10, 1)   # convert to 0-100
        feedbacks = [r.get("feedback", "") for r in results]
        return score, feedbacks
    except Exception:
        return 50.0, ["Could not evaluate."] * 3


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@verify_bp.route("/start", methods=["POST"])
def start_verification():
    """
    Generates a live coding task for a candidate.

    Body (JSON):
    {
        "claimed_skills":   ["python", "data structures", "sql"],
        "difficulty":       "medium",
        "candidate_email":  "dev@example.com"
    }
    """
    data       = request.json or {}
    skills     = data.get("claimed_skills", ["python"])
    difficulty = data.get("difficulty",     "medium")
    email      = data.get("candidate_email","")

    task = generate_coding_task(skills, difficulty)

    # Store session record
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO coding_verifications
            (candidate_email, claimed_skills, task_title, task_description,
             time_limit_sec, created_at)
        VALUES (?,?,?,?,?,?)
    """, (
        email,
        json.dumps(skills),
        task["title"],
        task["description"],
        task.get("time_limit_sec", 1200),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    verify_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return jsonify({
        "verify_id":    verify_id,
        "task": {
            "title":        task["title"],
            "description":  task["description"],
            "starter_code": task["starter_code"],
            "time_limit_sec": task.get("time_limit_sec", 1200),
        },
        "telemetry_fields": [
            "time_taken_sec", "keystrokes_per_min",
            "pause_count", "error_events", "tab_switches"
        ],
        "instruction": "Complete the task and submit your code along with the telemetry data collected by the browser.",
    })


@verify_bp.route("/submit", methods=["POST"])
def submit_code():
    """
    Candidate submits their code + browser telemetry.
    Returns code score, behavior score, and 3 follow-up questions.

    Body (JSON):
    {
        "verify_id":    1,
        "code":         "def solution(...):\\n    ...",
        "telemetry": {
            "time_taken_sec":     540,
            "keystrokes_per_min": 45,
            "pause_count":        2,
            "error_events":       3,
            "tab_switches":       0
        }
    }
    """
    data      = request.json or {}
    verify_id = data.get("verify_id")
    code      = data.get("code", "").strip()
    telemetry = data.get("telemetry", {})

    if not verify_id or not code:
        return jsonify({"error": "verify_id and code are required."}), 400

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM coding_verifications WHERE id = ?", (verify_id,)
    ).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Verification session not found."}), 404

    task = {
        "title":       row["task_title"],
        "description": row["task_description"],
        "hint_trap":   "",
    }
    time_limit = row["time_limit_sec"]

    # ── Evaluate code ──
    code_score, code_feedback = evaluate_submitted_code(task, code)

    # ── Score behavior ──
    behavior_score, behavior_breakdown = score_behavior(telemetry, time_limit)

    # ── Generate follow-up Qs ──
    followup_qs = generate_followup_questions(task, code)

    # ── Update DB ──
    conn.execute("""
        UPDATE coding_verifications SET
            submitted_code     = ?,
            time_taken_sec     = ?,
            keystrokes_per_min = ?,
            pause_count        = ?,
            error_events       = ?,
            tab_switches       = ?,
            code_score         = ?,
            behavior_score     = ?,
            followup_questions = ?
        WHERE id = ?
    """, (
        code,
        telemetry.get("time_taken_sec",     0),
        telemetry.get("keystrokes_per_min", 0),
        telemetry.get("pause_count",        0),
        telemetry.get("error_events",       0),
        telemetry.get("tab_switches",       0),
        code_score,
        behavior_score,
        json.dumps(followup_qs),
        verify_id,
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "verify_id":        verify_id,
        "code_score":       code_score,
        "code_feedback":    code_feedback,
        "behavior": {
            "score":     behavior_score,
            "breakdown": behavior_breakdown,
        },
        "flags": behavior_breakdown.get("flags", {}),
        "followup_questions": followup_qs,
        "next_step": "Answer the 3 follow-up questions and POST to /verify/answer_followup",
    })


@verify_bp.route("/answer_followup", methods=["POST"])
def answer_followup():
    """
    Candidate answers the 3 follow-up questions about their own code.
    Returns final verification score + verdict.

    Body (JSON):
    {
        "verify_id": 1,
        "answers": ["My set handles duplicates by...", "Empty list returns [].", "I'd use a hash map..."]
    }
    """
    data      = request.json or {}
    verify_id = data.get("verify_id")
    answers   = data.get("answers", [])

    if not verify_id or not answers:
        return jsonify({"error": "verify_id and answers are required."}), 400

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM coding_verifications WHERE id = ?", (verify_id,)
    ).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Verification session not found."}), 404

    questions = json.loads(row["followup_questions"] or "[]")
    code      = row["submitted_code"] or ""
    code_score    = row["code_score"]    or 0
    behavior_score = row["behavior_score"] or 0

    # ── Evaluate follow-up answers ──
    followup_score, feedbacks = evaluate_followup_answers(code, questions, answers)

    # ── Final composite score ──
    final_score = round(
        code_score     * 0.40 +
        behavior_score * 0.30 +
        followup_score * 0.30,
        1
    )

    # ── Verdict ──
    flags = {}
    if   final_score >= 80: verdict = "✅ VERIFIED — Strong evidence of genuine proficiency"
    elif final_score >= 65: verdict = "🟡 LIKELY GENUINE — Minor concerns; recommend interview"
    elif final_score >= 45: verdict = "🟠 UNCERTAIN — Claims may exceed demonstrated ability"
    else:                   verdict = "🔴 NOT VERIFIED — Significant gap between claims and performance"

    # ── Update DB ──
    conn.execute("""
        UPDATE coding_verifications SET
            followup_answers = ?, followup_score = ?,
            final_score = ?, verdict = ?
        WHERE id = ?
    """, (json.dumps(answers), followup_score, final_score, verdict, verify_id))
    conn.commit()
    conn.close()

    return jsonify({
        "verify_id":      verify_id,
        "final_score":    final_score,
        "verdict":        verdict,
        "score_breakdown": {
            "code_quality":   f"{code_score:.1f}  (40%)",
            "behavior":       f"{behavior_score:.1f}  (30%)",
            "followup_qa":    f"{followup_score:.1f}  (30%)",
        },
        "per_answer_feedback": [
            {"question": q, "answer": a, "feedback": f}
            for q, a, f in zip(questions, answers, feedbacks)
        ],
        "hr_recommendation": (
            "Shortlist for technical interview."       if final_score >= 70 else
            "Proceed with caution — verify in person." if final_score >= 50 else
            "Do not shortlist based on claimed skills."
        ),
    })


# ──────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL FPS ENGINE
# ──────────────────────────────────────────────────────────────────────────────
#
# What the judge meant by "FPS":
#   Every 10 seconds = one "frame".  The browser silently records what happened
#   in that window.  At the end we send the full timeline to AI which reads it
#   like a video — frame by frame — and decides: real coder or not?
#
# Browser must send these fields per frame (collected by the JS snippet below):
#   {
#     "t":         30,          # seconds elapsed at this frame
#     "keys":      42,          # keystrokes in this 10-sec window
#     "chars_net": 38,          # net new characters added (detects paste spikes)
#     "mouse":     true,        # any mouse movement?
#     "focused":   true,        # was editor tab focused?
#     "errors":    1,           # syntax errors shown by editor in this window
#     "paste":     false        # did a Ctrl+V / paste event fire?
#   }
#
# JS snippet to paste into the coding challenge page:
# ──────────────────────────────────────────────────
# const FPS_INTERVAL = 10000; // 10 seconds
# let frames = [], startTime = Date.now(), keys = 0, errors = 0, paste = false, lastLen = 0;
# document.addEventListener('keydown', () => keys++);
# document.addEventListener('paste',   () => { paste = true; });
# CodeMirror.on('change', (cm) => { errors = cm.state.lint?.marked?.length || 0; });
# setInterval(() => {
#   const code = editor.getValue();
#   frames.push({
#     t: Math.round((Date.now() - startTime) / 1000),
#     keys, chars_net: code.length - lastLen,
#     mouse: mouseMoved, focused: document.hasFocus(),
#     errors, paste
#   });
#   keys = 0; errors = 0; paste = false; mouseMoved = false;
#   lastLen = code.length;
# }, FPS_INTERVAL);
# // On submit → send frames to: POST /verify/fps_report
# ──────────────────────────────────────────────────────────────────────────────


def analyse_fps_timeline(frames: list[dict], task_title: str) -> dict:
    """
    Sends the full behavioral timeline to AI for pattern analysis.
    Returns a pattern report with authenticity signal.
    """
    # Build a compact readable timeline string for the AI
    lines = []
    for f in frames:
        flag = ""
        if f.get("paste"):               flag += "[PASTE] "
        if f.get("chars_net", 0) > 200:  flag += "[BULK-ADD] "
        if not f.get("focused"):         flag += "[UNFOCUSED] "
        lines.append(
            f"t={f.get('t',0):>4}s | keys={f.get('keys',0):>3} | "
            f"net_chars={f.get('chars_net',0):>5} | "
            f"errors={f.get('errors',0)} | "
            f"mouse={'Y' if f.get('mouse') else 'N'} | "
            f"focus={'Y' if f.get('focused', True) else 'N'} {flag}"
        )
    timeline_str = "\n".join(lines)

    prompt = f"""You are an AI proctoring analyst reviewing a candidate's live coding session
for the task: "{task_title}".

Below is a behavioral timeline sampled every 10 seconds (one frame per row).
Columns: elapsed time | keystrokes in window | net characters added |
         syntax errors | mouse active | editor focused | flags.

TIMELINE:
{timeline_str}

Analyse this timeline for human vs fabricated coding evidence.

Look for these RED FLAGS:
- Paste event OR chars_net suddenly jumps by >150 in one frame (copy-paste of solution)
- Long stretches of 0 keystrokes while time passes (looking up answer elsewhere)
- Editor loses focus repeatedly   (reading another tab)
- Perfectly smooth, no errors at all (suspiciously clean — real coders make typos)
- Code appears in one frame with no gradual build-up

And these GREEN signals:
- Gradual increase in net chars frame by frame (incremental development)
- Some syntax errors early that reduce over time (iterative debugging)
- Consistent keystroke rate (20-80 per 10-sec window)
- Brief pauses then activity resumes (thinking then coding)

Return ONLY this JSON:
{{
  "authenticity_signal": "genuine | suspicious | fabricated",
  "confidence": 0.0,
  "red_flags":  ["flag1", "flag2"],
  "green_signals": ["signal1"],
  "peak_productivity_window": "t=Xs to t=Ys",
  "analysis_summary": "2-3 sentence plain-English summary for HR"
}}
"""
    try:
        resp = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[Verify-FPS] Analysis error: {e}")
        return {
            "authenticity_signal": "unknown",
            "confidence":          0.5,
            "red_flags":           [],
            "green_signals":       [],
            "peak_productivity_window": "N/A",
            "analysis_summary":    "Could not analyse behavioral timeline.",
        }


def _compute_fps_metrics(frames: list[dict]) -> dict:
    """Derives summary stats from the raw frame list."""
    if not frames:
        return {}
    total_keys  = sum(f.get("keys",     0) for f in frames)
    paste_count = sum(1 for f in frames if f.get("paste"))
    unfocused   = sum(1 for f in frames if not f.get("focused", True))
    bulk_frames = sum(1 for f in frames if f.get("chars_net", 0) > 150)
    zero_frames = sum(1 for f in frames if f.get("keys", 0) == 0)
    max_chars_spike = max((f.get("chars_net", 0) for f in frames), default=0)
    duration_sec    = frames[-1].get("t", 0) if frames else 0
    avg_kps = round(total_keys / max(duration_sec, 1), 2)   # keystrokes/sec

    return {
        "total_frames":        len(frames),
        "duration_sec":        duration_sec,
        "total_keystrokes":    total_keys,
        "avg_keystrokes_sec":  avg_kps,
        "paste_events":        paste_count,
        "unfocused_frames":    unfocused,
        "bulk_add_frames":     bulk_frames,
        "idle_frames":         zero_frames,
        "max_chars_spike":     max_chars_spike,
    }


@verify_bp.route("/fps_report", methods=["POST"])
def fps_report():
    """
    Receives the behavioral FPS timeline from the browser and
    returns an AI-powered coding authenticity analysis.

    Body (JSON):
    {
        "verify_id": 1,
        "frames": [
            {"t": 10, "keys": 35, "chars_net": 30, "mouse": true,
             "focused": true, "errors": 2, "paste": false},
            {"t": 20, "keys": 42, "chars_net": 38, "mouse": true,
             "focused": true, "errors": 1, "paste": false},
            ...
        ]
    }
    """
    data      = request.json or {}
    verify_id = data.get("verify_id")
    frames    = data.get("frames", [])

    if not verify_id:
        return jsonify({"error": "verify_id is required."}), 400
    if not frames:
        return jsonify({"error": "frames list is empty — nothing to analyse."}), 400

    # Fetch task title for context
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT task_title FROM coding_verifications WHERE id = ?", (verify_id,)
    ).fetchone()
    conn.close()
    task_title = row["task_title"] if row else "Unknown Task"

    # Compute summary metrics
    metrics = _compute_fps_metrics(frames)

    # AI timeline analysis
    analysis = analyse_fps_timeline(frames, task_title)

    # Map signal → HR action
    signal = analysis.get("authenticity_signal", "unknown")
    hr_note = {
        "genuine":    "✅ Coding pattern is consistent with genuine work. Proceed.",
        "suspicious": "⚠️  Some anomalies detected. Recommend a follow-up live interview.",
        "fabricated": "🔴 Strong evidence of copy-paste or external assistance. Flag for review.",
    }.get(signal, "❓ Inconclusive — review manually.")

    return jsonify({
        "verify_id":  verify_id,
        "task":       task_title,
        "metrics":    metrics,
        "fps_analysis": analysis,
        "hr_note":    hr_note,
        "what_is_fps": (
            "Each 'frame' = a 10-second snapshot of the candidate's coding behavior. "
            "AI reads the full timeline like a video to detect paste events, idle periods, "
            "and unnatural typing patterns — proving or disproving coding authenticity."
        ),
    })

