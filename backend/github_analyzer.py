# github_analyzer.py

import os
import re
import json
import base64
import datetime
import requests
import sqlite3

from flask import Blueprint, request, jsonify
from openai import OpenAI

# ─────────────────────────────────────────────
github_bp = Blueprint("github", __name__, url_prefix="/github")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GROQ_API_KEY:
    raise ValueError("Missing required environment variable: GROQ_API_KEY")

_ai = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"

DB_PATH = "placement.db"

# ─────────────────────────────────────────────
# DB SETUP (UPDATED)
# ─────────────────────────────────────────────

def _init_github_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS github_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_email TEXT,
            repo_url TEXT,
            repo_name TEXT,
            owner TEXT,
            stars INTEGER,
            forks INTEGER,
            total_commits INTEGER,
            languages TEXT,
            ai_questions TEXT,
            candidate_answers TEXT,
            authenticity_score REAL,
            answer_feedback TEXT,
            commit_quality TEXT,
            ownership TEXT,
            code_sample TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

_init_github_table()

# ─────────────────────────────────────────────
# GITHUB HELPERS
# ─────────────────────────────────────────────

_GH_HEADERS = {
    "Accept": "application/vnd.github+json"
}
if GITHUB_TOKEN:
    _GH_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

def _gh_get(url):
    try:
        r = requests.get(url, headers=_GH_HEADERS, timeout=8)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def parse_repo_url(url):
    url = url.strip()
    if "github.com/" in url:
        parts = url.split("github.com/")[1].split("/")
        if len(parts) >= 2:
            return parts[0], parts[1].replace(".git", "")
        elif len(parts) >= 1:
            return parts[0], None
    return url.replace("/", ""), None

# ─────────────────────────────────────────────
# FETCH REPO DATA (UPDATED)
# ─────────────────────────────────────────────

def fetch_repo_data(owner, repo=None):
    if not repo:
        # Fetch user's top projects if only owner is provided
        base_url = f"https://api.github.com/users/{owner}/repos?sort=updated&per_page=3"
        repos_raw = _gh_get(base_url)
        if not isinstance(repos_raw, list):
            repos_raw = []
        
        total_stars = 0
        total_forks = 0
        all_langs = {}
        commits_raw = []
        contributors_raw = []
        code_samples = ""
        repo_names = []
        
        for r in repos_raw:
            repo_names.append(r.get("name", ""))
            total_stars += r.get("stargazers_count", 0)
            total_forks += r.get("forks_count", 0)
            
            repo_base = f"https://api.github.com/repos/{owner}/{r.get('name')}"
            
            # Languages
            langs_raw = _gh_get(f"{repo_base}/languages")
            if isinstance(langs_raw, dict):
                for k, v in langs_raw.items():
                    all_langs[k] = all_langs.get(k, 0) + v
                    
            # Commits
            c_raw = _gh_get(f"{repo_base}/commits?per_page=10")
            if isinstance(c_raw, list):
                commits_raw.extend(c_raw)
                
            # Contributors
            contrib_raw = _gh_get(f"{repo_base}/contributors?per_page=3")
            if isinstance(contrib_raw, list):
                contributors_raw.extend(contrib_raw)
                
            # Code Sample
            readme = _gh_get(f"{repo_base}/readme")
            if readme and readme.get("content"):
                code_samples += f"\n--- Repo: {r.get('name')} ---\n"
                code_samples += base64.b64decode(readme["content"]).decode("utf-8")[:800]
                
        repo = ", ".join(repo_names)
        
        langs_dict = {k: v for k, v in all_langs.items() if isinstance(v, (int, float))}
        total = sum(langs_dict.values()) or 1
        languages = {k: round(v/total*100, 1) for k, v in langs_dict.items()}
        
        info = {
            "description": f"User {owner} projects combining {len(repos_raw)} recently updated repositories.",
            "stargazers_count": total_stars,
            "forks_count": total_forks
        }
        
    else:
        # Single Repo
        base = f"https://api.github.com/repos/{owner}/{repo}"
        info = _gh_get(base)
        if not isinstance(info, dict):
            info = {}

        commits_raw = _gh_get(f"{base}/commits?per_page=30")
        if not isinstance(commits_raw, list):
            commits_raw = []

        langs_raw = _gh_get(f"{base}/languages")
        if not isinstance(langs_raw, dict):
            langs_raw = {}
        langs_dict = {k: v for k, v in langs_raw.items() if isinstance(v, (int, float))}
        total = sum(langs_dict.values()) or 1
        languages = {k: round(v/total*100, 1) for k, v in langs_dict.items()}

        contributors_raw = _gh_get(f"{base}/contributors")
        if not isinstance(contributors_raw, list):
            contributors_raw = []

        code_samples = ""
        readme = _gh_get(f"{base}/readme")
        if readme and readme.get("content"):
            code_samples = base64.b64decode(readme["content"]).decode("utf-8")[:2000]

    # Shared processing
    messages = [c.get("commit", {}).get("message", "") for c in commits_raw[:20] if isinstance(c, dict)]
    meaningful = sum(1 for m in messages if len(m.split()) >= 3)
    commit_quality = "good" if meaningful >= 8 else "average" if meaningful >= 4 else "poor"

    contributors = [c.get("login", "").lower() for c in contributors_raw[:15] if isinstance(c, dict)]
    commit_emails = [
        c.get("commit", {}).get("author", {}).get("email", "").lower()
        for c in commits_raw if isinstance(c, dict)
    ]

    return {
        "owner": owner,
        "repo": repo,
        "stars": info.get("stargazers_count", 0),
        "forks": info.get("forks_count", 0),
        "description": info.get("description", ""),
        "languages": languages,
        "commit_messages": messages,
        "commit_quality": commit_quality,
        "contributors": contributors,
        "commit_emails": commit_emails,
        "total_commits": len(commits_raw),
        "code_sample": code_samples
    }

# ─────────────────────────────────────────────
# OWNERSHIP CHECK (NEW 🔥)
# ─────────────────────────────────────────────

def check_ownership(email, contributors, commit_emails):
    if not email:
        return "unknown"

    email = email.lower()

    if email in commit_emails:
        return "verified"
    elif contributors:
        return "partial"
    else:
        return "not_verified"

# ─────────────────────────────────────────────
# AI FUNCTIONS
# ─────────────────────────────────────────────

def generate_repo_questions(repo_data):
    prompt = f"""
Projects: {repo_data['owner']}/{repo_data['repo']}
Desc: {repo_data['description']}
Code Samples: {repo_data['code_sample'][:2500]}

Generate exactly 4 technical interview questions based strictly on the code structures, languages, and patterns in these projects as a raw JSON list of strings. Do not include markdown formatting.
Make them challenging and focused on the user's specific implementations.
Example: ["Question 1?", "Question 2?", ...]
"""
    try:
        r = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = r.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Error parsing questions: {e}")
        return ["Explain the overall architecture?", "What was the hardest bug you faced?", "Why did you choose this tech stack?", "How would you improve this codebase?"]

def evaluate_repo_answers(repo_data, questions, answers):
    prompt = f"""
Code: {repo_data['code_sample'][:800]}

Q&A:
{list(zip(questions, answers))}

Based on how authentically and accurately the user answered the questions referencing the code, return ONLY a raw JSON array of objects. No markdown formatting.
Format:
[{{"answer_score": 85, "feedback": "Good explanation"}}]
"""
    try:
        r = _ai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = r.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw)
        scores = [x.get("answer_score", 50) for x in data]
        return sum(scores)/max(len(scores), 1), data
    except Exception as e:
        print(f"Error parsing evaluation: {e}")
        return 50, [{"answer_score": 50, "feedback": "Could not parse evaluation."} for _ in questions]

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@github_bp.route("/dashboard", methods=["GET"])
def github_dashboard_view():
    from flask import render_template
    return render_template("github_dashboard.html")

@github_bp.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    url = data.get("repo_url")
    email = data.get("candidate_email", "")

    parsed = parse_repo_url(url)
    if not parsed:
        return jsonify({"error": "Invalid URL or Username"}), 400

    owner, repo = parsed
    repo_data = fetch_repo_data(owner, repo)

    questions = generate_repo_questions(repo_data)

    ownership = check_ownership(
        email,
        repo_data["contributors"],
        repo_data["commit_emails"]
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO github_analyses 
        (candidate_email, repo_url, repo_name, owner, stars, forks,
         total_commits, languages, ai_questions, commit_quality,
         ownership, code_sample, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        email, url, repo, owner,
        repo_data["stars"], repo_data["forks"],
        repo_data["total_commits"],
        json.dumps(repo_data["languages"]),
        json.dumps(questions),
        repo_data["commit_quality"],
        ownership,
        repo_data["code_sample"],
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    analysis_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return jsonify({
        "analysis_id": analysis_id,
        "ownership": ownership,
        "questions": questions
    })

# ─────────────────────────────────────────────

@github_bp.route("/evaluate_answers", methods=["POST"])
def evaluate():
    data = request.json
    analysis_id = data["analysis_id"]
    answers = data["answers"]

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM github_analyses WHERE id=?",
        (analysis_id,)
    ).fetchone()

    questions = json.loads(row["ai_questions"])

    repo_data = {
        "owner": row["owner"],
        "repo": row["repo_name"],
        "languages": json.loads(row["languages"]),
        "code_sample": row["code_sample"]
    }

    score, feedback = evaluate_repo_answers(repo_data, questions, answers)

    conn.execute("""
        UPDATE github_analyses
        SET candidate_answers=?, authenticity_score=?, answer_feedback=?
        WHERE id=?
    """, (
        json.dumps(answers),
        score,
        json.dumps(feedback),
        analysis_id
    ))
    conn.commit()
    conn.close()

    return jsonify({
        "score": score,
        "feedback": feedback
    })

# ─────────────────────────────────────────────

@github_bp.route("/report/<int:id>", methods=["GET"])
def report(id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM github_analyses WHERE id=?", (id,)).fetchone()
    conn.close()

    return jsonify(dict(row))

