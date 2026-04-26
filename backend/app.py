from backend.speech_service import transcribe_audio
from dotenv import load_dotenv
import os
import re
import time
import json
import random
import tempfile
import subprocess
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
from openai import OpenAI
import sqlite3
import datetime

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-key-change-in-production")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB limit
os.makedirs("uploads", exist_ok=True)

# ── Placement Engine Blueprint ──
from backend.placement_engine import placement_bp

app.register_blueprint(placement_bp)

# ── GitHub Intelligence Blueprint ──
from backend.github_analyzer import github_bp

app.register_blueprint(github_bp)

# ── Live Coding Verifier Blueprint ──
from backend.coding_verifier import verify_bp

app.register_blueprint(verify_bp)

# ── Static HTML Pages Routing ──
@app.route('/contact.html')
@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/cloud.html')
@app.route('/cloud')
def cloud_page():
    return render_template('cloud.html')



def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS interviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        resume_score REAL,
        cognitive_score REAL,
        coding_score REAL,
        communication_score REAL,
        overall_score REAL,
        cheating_attempts INTEGER,
        feedback_summary TEXT
    )
    """
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        name TEXT,
        email TEXT,
        category TEXT,
        message TEXT,
        rating INTEGER
    )
    """
    )
    conn.execute('''CREATE TABLE IF NOT EXISTS company_simulations (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, candidate_name TEXT, company TEXT, role TEXT, score INTEGER, total INTEGER, percentage REAL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS user_xp (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        username TEXT,
        xp INTEGER DEFAULT 0,
        level INTEGER DEFAULT 1,
        streak INTEGER DEFAULT 0,
        last_activity_date TEXT,
        total_interviews INTEGER DEFAULT 0,
        best_score REAL DEFAULT 0,
        target_role TEXT DEFAULT "Software Engineer"
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS xp_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        date TEXT,
        activity TEXT,
        xp_earned INTEGER,
        score REAL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS roadmap_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        role TEXT,
        week INTEGER,
        task_idx INTEGER,
        completed INTEGER DEFAULT 0,
        date TEXT
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS video_interviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        date TEXT,
        question TEXT,
        transcript TEXT,
        score REAL,
        feedback TEXT,
        confidence_score REAL
    )''')
    conn.commit()
    conn.close()


init_db()

# =========================
# CONFIG & AI INIT
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing required environment variable: GROQ_API_KEY")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"

# Evaluation Weightage
WEIGHTS = {"resume": 0.30, "coding": 0.45, "cognitive": 0.25}
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# =========================
# JOB ROLES DATABASE
# =========================
# Each role has core_skills (must-have) and good_to_have (bonus)
JOB_ROLES_DB = {
    "Data Scientist": {
        "icon": "🧪",
        "core": [
            "python",
            "machine learning",
            "statistics",
            "data analysis",
            "pandas",
            "numpy",
            "scikit-learn",
        ],
        "good_to_have": [
            "deep learning",
            "tensorflow",
            "pytorch",
            "sql",
            "matplotlib",
            "seaborn",
            "r",
            "spark",
        ],
        "desc": "Build predictive models and extract insights from complex datasets.",
    },
    "Machine Learning Engineer": {
        "icon": "🤖",
        "core": [
            "python",
            "machine learning",
            "tensorflow",
            "pytorch",
            "mlops",
            "model deployment",
        ],
        "good_to_have": [
            "docker",
            "kubernetes",
            "aws",
            "fastapi",
            "scikit-learn",
            "spark",
            "airflow",
        ],
        "desc": "Design, train, and deploy ML models at production scale.",
    },
    "AI/ML Researcher": {
        "icon": "🔬",
        "core": [
            "python",
            "deep learning",
            "pytorch",
            "research",
            "nlp",
            "computer vision",
        ],
        "good_to_have": [
            "tensorflow",
            "transformers",
            "hugging face",
            "cuda",
            "latex",
            "mathematics",
        ],
        "desc": "Advance the state-of-the-art in artificial intelligence and machine learning.",
    },
    "Full Stack Developer": {
        "icon": "🌐",
        "core": ["javascript", "react", "node.js", "html", "css", "rest api", "sql"],
        "good_to_have": [
            "typescript",
            "next.js",
            "mongodb",
            "docker",
            "graphql",
            "redis",
            "aws",
        ],
        "desc": "Build end-to-end web applications from UI to server to database.",
    },
    "Backend Developer": {
        "icon": "⚙️",
        "core": [
            "python",
            "java",
            "node.js",
            "rest api",
            "sql",
            "databases",
            "microservices",
        ],
        "good_to_have": [
            "docker",
            "kubernetes",
            "redis",
            "rabbitmq",
            "aws",
            "fastapi",
            "django",
            "spring boot",
        ],
        "desc": "Design scalable server-side systems, APIs, and data pipelines.",
    },
    "Frontend Developer": {
        "icon": "🎨",
        "core": ["javascript", "react", "html", "css", "typescript", "ui/ux"],
        "good_to_have": [
            "vue.js",
            "angular",
            "next.js",
            "tailwind",
            "figma",
            "webpack",
            "testing",
        ],
        "desc": "Create beautiful, responsive, and accessible user interfaces.",
    },
    "DevOps Engineer": {
        "icon": "🚀",
        "core": ["docker", "kubernetes", "ci/cd", "linux", "terraform", "aws"],
        "good_to_have": [
            "jenkins",
            "ansible",
            "prometheus",
            "grafana",
            "azure",
            "gcp",
            "bash",
            "python",
        ],
        "desc": "Automate infrastructure, deployments, and ensure system reliability.",
    },
    "Cloud Engineer": {
        "icon": "☁️",
        "core": [
            "aws",
            "azure",
            "gcp",
            "cloud architecture",
            "terraform",
            "networking",
        ],
        "good_to_have": [
            "docker",
            "kubernetes",
            "python",
            "security",
            "cost optimization",
            "serverless",
        ],
        "desc": "Architect and manage scalable cloud infrastructure solutions.",
    },
    "Data Engineer": {
        "icon": "🔧",
        "core": ["python", "sql", "spark", "data pipelines", "etl", "kafka"],
        "good_to_have": [
            "airflow",
            "dbt",
            "snowflake",
            "aws",
            "bigquery",
            "hadoop",
            "scala",
        ],
        "desc": "Build and maintain robust data infrastructure and pipeline systems.",
    },
    "Cybersecurity Analyst": {
        "icon": "🛡️",
        "core": [
            "networking",
            "security",
            "penetration testing",
            "linux",
            "firewalls",
            "siem",
        ],
        "good_to_have": [
            "python",
            "ethical hacking",
            "cryptography",
            "incident response",
            "soc",
            "cloud security",
        ],
        "desc": "Protect systems and networks from cyber threats and vulnerabilities.",
    },
    "Android Developer": {
        "icon": "📱",
        "core": ["kotlin", "java", "android", "xml", "rest api", "android studio"],
        "good_to_have": [
            "jetpack compose",
            "mvvm",
            "firebase",
            "room db",
            "coroutines",
            "google play",
        ],
        "desc": "Build native Android mobile applications.",
    },
    "iOS Developer": {
        "icon": "🍎",
        "core": ["swift", "objective-c", "xcode", "ios", "uikit", "swiftui"],
        "good_to_have": [
            "cocoapods",
            "core data",
            "combine",
            "testflight",
            "firebase",
            "rest api",
        ],
        "desc": "Build native iOS applications for iPhone and iPad.",
    },
    "Embedded Systems Engineer": {
        "icon": "💡",
        "core": [
            "c",
            "c++",
            "microcontrollers",
            "rtos",
            "embedded linux",
            "uart",
            "spi",
            "i2c",
        ],
        "good_to_have": [
            "python",
            "arduino",
            "raspberry pi",
            "fpga",
            "verilog",
            "pcb design",
            "can bus",
        ],
        "desc": "Develop firmware and low-level software for hardware devices.",
    },
    "Robotics Engineer": {
        "icon": "🦾",
        "core": ["ros", "python", "c++", "kinematics", "control systems", "sensors"],
        "good_to_have": [
            "gazebo",
            "slam",
            "computer vision",
            "opencv",
            "matlab",
            "mechanical design",
        ],
        "desc": "Design and program autonomous robotic systems.",
    },
    "Data Analyst": {
        "icon": "📊",
        "core": ["python", "sql", "excel", "data analysis", "tableau", "power bi"],
        "good_to_have": [
            "r",
            "statistics",
            "pandas",
            "numpy",
            "visualization",
            "business intelligence",
        ],
        "desc": "Analyse business data to drive strategic decisions.",
    },
    "NLP Engineer": {
        "icon": "💬",
        "core": ["python", "nlp", "transformers", "hugging face", "spacy", "nltk"],
        "good_to_have": [
            "pytorch",
            "tensorflow",
            "bert",
            "gpt",
            "text classification",
            "named entity recognition",
        ],
        "desc": "Build language understanding and text processing AI systems.",
    },
    "Computer Vision Engineer": {
        "icon": "👁️",
        "core": [
            "python",
            "computer vision",
            "opencv",
            "deep learning",
            "pytorch",
            "image processing",
        ],
        "good_to_have": [
            "yolo",
            "tensorflow",
            "cnn",
            "object detection",
            "segmentation",
            "cuda",
        ],
        "desc": "Build systems that can understand and interpret visual data.",
    },
    "Blockchain Developer": {
        "icon": "⛓️",
        "core": [
            "solidity",
            "ethereum",
            "web3",
            "smart contracts",
            "blockchain",
            "javascript",
        ],
        "good_to_have": ["hardhat", "truffle", "ipfs", "defi", "nft", "rust", "python"],
        "desc": "Build decentralised applications and smart contract systems.",
    },
    "Game Developer": {
        "icon": "🎮",
        "core": ["unity", "c#", "unreal engine", "c++", "game design", "physics"],
        "good_to_have": [
            "blender",
            "opengl",
            "vulkan",
            "shader programming",
            "networking",
            "python",
        ],
        "desc": "Design and develop interactive games across platforms.",
    },
    "QA / Test Engineer": {
        "icon": "✅",
        "core": [
            "manual testing",
            "automation testing",
            "selenium",
            "python",
            "java",
            "test cases",
        ],
        "good_to_have": [
            "pytest",
            "cypress",
            "jira",
            "rest api testing",
            "postman",
            "performance testing",
            "ci/cd",
        ],
        "desc": "Ensure software quality through rigorous testing and quality engineering.",
    },
}

# =========================
# CORE UTILITIES
# =========================


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_path):
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(
                    [p.extract_text() for p in reader.pages if p.extract_text()]
                )
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text


def save_session_to_db(
    resume_score,
    cognitive_score,
    coding_score,
    communication_score,
    overall_score,
    cheating_attempts,
    feedback_summary,
):
    try:
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO interviews (date, resume_score, cognitive_score, coding_score, communication_score, overall_score, cheating_attempts, feedback_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                current_date,
                resume_score,
                cognitive_score,
                coding_score,
                communication_score,
                overall_score,
                cheating_attempts,
                feedback_summary,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")


# =========================
# ADAPTIVE AI ENGINE
# =========================


def analyze_resume_ai(text):
    """Classifies candidate and extracts skills for adaptive logic."""
    prompt = f"""
Analyze this resume: {text[:2000]}
1. Identify if the profile is 'Technical' (Software/Data/Eng) or 'Non-Technical'.
2. Extract key skills as a comma-separated list.
Return Format: Type: [Type], Skills: [List]
"""
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2
        )
        content = response.choices[0].message.content
        is_tech = (
            "Technical" in content and "Non-Technical" not in content.split("Type:")[1]
        )
        skills = content.split("Skills:")[1].strip() if "Skills:" in content else ""
        return skills, is_tech
    except:
        return "General", False


def deep_analyze_resume(text):
    """
    Deep resume parser: separates project-verified skills (exact match) from
    skills only listed in the Skills section (partial match), and predicts the
    best-fit job role with a confidence score.
    """
    prompt = f"""
You are an expert career analyst and resume parser. Analyze the following resume text deeply.

Resume:
{text[:4000]}

Your task:
1. Identify the single best-fit 'predicted_role' — the most specific job title this candidate is suited for.
   Must be one of the job titles listed here: {', '.join(JOB_ROLES_DB.keys())}.
2. Extract 'verified_skills': skills that appear INSIDE project descriptions, work experience, or achievements.
   These are skills the candidate has ACTUALLY USED and can be demonstrated.
3. Extract 'partial_skills': skills listed ONLY in a 'Skills' or 'Technologies' section but NOT backed by a project.
4. Write 1-2 sentences in 'profile_summary' about the candidate's strongest areas.
5. Give a 'confidence_score' between 0.0 and 1.0 for how well the resume matches the predicted role.

Rules:
- Normalize all skill names to lowercase (e.g. 'Python' → 'python', 'Machine Learning' → 'machine learning').
- Be precise. Do not infer skills; only report what is stated.
- Return ONLY valid JSON. No markdown, no explanation.

Format:
{{
    "predicted_role": "string",
    "verified_skills": ["skill1", "skill2"],
    "partial_skills": ["skill3", "skill4"],
    "profile_summary": "string",
    "confidence_score": 0.0
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        # Normalise
        data["verified_skills"] = [
            s.lower().strip() for s in data.get("verified_skills", [])
        ]
        data["partial_skills"] = [
            s.lower().strip() for s in data.get("partial_skills", [])
        ]
        return data
    except Exception as e:
        print(f"Deep resume analysis error: {e}")
        return {
            "predicted_role": "Full Stack Developer",
            "verified_skills": [],
            "partial_skills": [],
            "profile_summary": "Could not parse resume details.",
            "confidence_score": 0.0,
        }


def compute_job_match(verified_skills, partial_skills, role_data):
    """
    Calculates a match score for a single job role.
    Scoring weights:
      - Exact match  (verified in project)  → 1.0 per skill
      - Partial match (skills-section only)  → 0.4 per skill
      - Good-to-have bonus                   → 0.15 per skill (normalised)
    Returns score 0–100, plus lists of matched / missing skills.
    """
    core = [s.lower() for s in role_data["core"]]
    good = [s.lower() for s in role_data["good_to_have"]]
    all_cand = set(verified_skills) | set(partial_skills)

    exact_hits = [s for s in core if s in verified_skills]
    partial_hits = [s for s in core if s in partial_skills and s not in verified_skills]
    missing = [s for s in core if s not in all_cand]
    bonus_hits = [s for s in good if s in all_cand]

    if not core:
        return 0, [], [], []

    core_score = (len(exact_hits) * 1.0 + len(partial_hits) * 0.4) / len(core)
    bonus_score = (len(bonus_hits) / len(good)) if good else 0

    # Weighted final: 85% core, 15% bonus
    raw_score = core_score * 85 + bonus_score * 15
    return round(min(raw_score, 100), 1), exact_hits, partial_hits, missing


def get_top_recommendations(analysis_data, top_n=5):
    """
    Scores every role in JOB_ROLES_DB and returns the top N by match score.
    The AI-predicted role is always shown at position 0 (highlighted).
    """
    verified = analysis_data.get("verified_skills", [])
    partial = analysis_data.get("partial_skills", [])
    predicted = analysis_data.get("predicted_role", "")

    results = []
    for role_name, role_data in JOB_ROLES_DB.items():
        score, exact_hits, partial_hits, missing = compute_job_match(
            verified, partial, role_data
        )
        results.append(
            {
                "title": role_name,
                "icon": role_data["icon"],
                "desc": role_data["desc"],
                "score": score,
                "exact_hits": exact_hits,
                "partial_hits": partial_hits,
                "missing": missing[:5],  # top 5 missing skills to learn
                "is_predicted": (role_name == predicted),
            }
        )

    # Sort by score descending
    results.sort(key=lambda x: (x["is_predicted"], x["score"]), reverse=True)

    # Ensure predicted role is always first
    pred_items = [r for r in results if r["is_predicted"]]
    rest = [r for r in results if not r["is_predicted"]]
    final = pred_items + rest
    return final[:top_n]


def generate_adaptive_questions(skills, is_tech):
    """Generates 5 questions based on resume strength."""
    role = "Software Engineer" if is_tech else "General Professional"
    prompt = f"As an interviewer for a {role} role, generate 5 completely unique and random questions based on these skills: {skills}. Make sure the questions vary significantly between different interviews. Return only the questions, one per line."
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9
    )
    return [
        q.strip()
        for q in response.choices[0].message.content.split("\n")
        if len(q) > 10
    ][:5]


def evaluate_answer_ai(question, answer):
    """Contextual AI evaluation with strictly grounded logic and robust parsing."""

    # SYSTEM PROMPT: Forces the AI to stay on topic and ignore previous contexts
    system_instruction = (
        "You are a technical interviewer. Evaluate the candidate's answer ONLY "
        "based on the provided question. If the answer is irrelevant, nonsense, "
        "or an error message, provide a score of 0."
    )

    user_prompt = f"""
Current Question: {question}
Candidate's Answer: {answer}

Evaluation Criteria:
1. Technical accuracy regarding the specific components mentioned in the question.
2. Clarity of the explanation.

Format your response EXACTLY as follows:
Score: [X]/10
Feedback: [Brief technical critique]
Followup: [One advanced technical question related ONLY to this topic if score > 7, else leave blank]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content

        # Robust parsing using Regular Expressions (Regex)
        score_match = re.search(r"Score:\s*(\d+)", content)
        score = int(score_match.group(1)) if score_match else 0

        feedback = "No feedback provided."
        if "Feedback:" in content:
            feedback = content.split("Feedback:")[1].split("Followup:")[0].strip()

        followup = ""
        if "Followup:" in content:
            followup = content.split("Followup:")[1].strip()

        return score, feedback, followup

    except Exception as e:
        print(f"AI Evaluation Error: {e}")
        return 0, "Technical error during evaluation.", ""


# =========================
# ROUTES
# =========================


@app.route("/")
def home():
    # Allow resetting session explicitly via a query param ?reset=1
    if request.args.get('reset'):
        session.clear()
    return render_template("index.html")


@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    resume_text = ""
    # 1. Handle File Upload
    if "resume" in request.files and request.files["resume"].filename != "":
        file = request.files["resume"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            resume_text = extract_text_from_file(path)
    # 2. Handle Text Paste
    elif request.form.get("resume_text"):
        resume_text = request.form.get("resume_text")

    if not resume_text:
        return redirect(url_for("home"))

    # --- Fast surface analysis (for questions) ---
    skills, is_tech = analyze_resume_ai(resume_text)
    # --- Deep analysis (for job recommendations) ---
    analysis_data = deep_analyze_resume(resume_text)

    session["is_tech"] = is_tech
    session["skills"] = skills
    session["analysis_data"] = analysis_data  # store for recommendations route
    session["resume_questions"] = generate_adaptive_questions(skills, is_tech)
    session["resume_idx"] = 0
    session["resume_score_total"] = 0
    session["cheating_attempts"] = 0

    return redirect(url_for("dashboard"))

@app.route("/resume_analyzer")
def resume_analyzer():
    """Displays the extracted resume profile & skills."""
    analysis_data = session.get("analysis_data")
    if not analysis_data:
        return redirect(url_for("dashboard"))
    
    return render_template(
        "resume_analyzer.html",
        skills=session.get("skills", "").split(","),
        analysis=analysis_data
    )

@app.route("/start_resume_interview")
def start_resume_interview():
    """Starts or restarts the mock resume interview."""
    session["resume_idx"] = 0
    session["resume_score_total"] = 0
    skills = session.get("skills", "")
    questions = session.get("resume_questions", [])
    
    if not questions:
        return redirect(url_for("dashboard"))

    return render_template(
        "resume_interview.html",
        question=questions[0],
        skills=skills.split(","),
    )


@app.route("/submit_resume_answer", methods=["POST"])
def submit_resume_answer():
    answer = request.form.get("answer")
    q_list = session.get("resume_questions")
    idx = session.get("resume_idx", 0)

    if not q_list:
        return jsonify({"redirect": url_for("home")})

    if idx >= len(q_list):
        return jsonify({"redirect": url_for("dashboard")})

    score, feedback, followup = evaluate_answer_ai(q_list[idx], answer)
    session["resume_score_total"] += score

    # AI Adaptive Follow-up Logic
    if followup and score > 7 and "followup_active" not in session:
        session["followup_active"] = True
        return jsonify(
            {"question": followup, "feedback": feedback, "transcript": answer}
        )

    session.pop("followup_active", None)
    session["resume_idx"] += 1

    if session["resume_idx"] >= len(q_list):
        session["resume_final_pct"] = (
            session["resume_score_total"] / (len(q_list) * 10)
        ) * 100
        return jsonify({"redirect": url_for("dashboard")})

    return jsonify(
        {
            "question": q_list[session["resume_idx"]],
            "feedback": feedback,
            "transcript": answer,
        }
    )


@app.route("/cognitive")
def cognitive_round():
    """Generates dynamic Aptitude and Engineering questions using Grok."""
    mode = request.args.get('mode')

    if mode == 'verbal':
        prompt = """
Generate 5 highly randomized multiple-choice questions for a 'Verbal Reasoning and Language Proficiency' test.
- 2 questions on Vocabulary (Synonyms, Antonyms, meaning).
- 2 questions on Grammar & Sentence Correction.
- 1 question on Reading Comprehension or Logical Deduction based on text.

Format the response as a valid JSON list of objects:
[
    {
    "question": "The actual question text",
    "options": ["Choice 1", "Choice 2", "Choice 3", "Choice 4"],
    "a": "The exact string of the correct answer"
    }
]
Return ONLY the JSON. No conversational text.
"""
    else:
        prompt = """
Generate 5 diverse, highly randomized multiple-choice questions for a technical recruitment 'Cognitive Phase'. Ensure these are different every time you are asked, selecting randomly from a large possible pool.
- 2 questions on Aptitude (Logic, Quantitative, or Critical Thinking).
- 3 questions on Core Engineering (Physics, Electronics, or Mechanical principles).

Format the response as a valid JSON list of objects:
[
    {
    "question": "The actual question text",
    "options": ["Choice 1", "Choice 2", "Choice 3", "Choice 4"],
    "a": "The exact string of the correct answer"
    }
]
Return ONLY the JSON. No conversational text.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9
        )

        # Extract and parse the JSON from Grok
        raw_content = response.choices[0].message.content.strip()
        # Clean up possible markdown code blocks if Grok includes them
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()

        dynamic_pool = json.loads(raw_content)
        session["cog_pool"] = dynamic_pool

    except Exception as e:
        print(f"Grok API Error: {e}")
        # Fallback to a hardcoded pool if the API fails
        dynamic_pool = [
            {
                "question": "A bat and ball cost $1.10. Bat is $1 more. Ball cost?",
                "options": ["0.10", "0.05", "0.01", "0.15"],
                "a": "0.05",
            },
            {
                "question": "Core Engineering: Unit of electrical resistance?",
                "options": ["Volt", "Ampere", "Ohm", "Watt"],
                "a": "Ohm",
            },
        ]
        session["cog_pool"] = dynamic_pool

    return render_template("cognitive.html", questions=dynamic_pool, mode=mode)


@app.route("/submit_cognitive", methods=["POST"])
def submit_cognitive():
    user_answers = request.json.get("answers", [])
    correct = 0
    pool = session.get("cog_pool", [])

    for i, q in enumerate(pool):
        if i < len(user_answers) and user_answers[i] == q["a"]:
            correct += 1

    score_pct = (correct / len(pool)) * 100 if pool else 0
    session["cognitive_score"] = score_pct

    return jsonify({"redirect": url_for("dashboard")})


@app.route("/coding")
def coding_round():
    is_tech = session.get("is_tech", False)
    skills = session.get("skills", "basic programming")
    
    prompt = f"Generate a single, precise technical coding challenge question suitable for a candidate with these skills: {skills}. Start directly with the question statement, make it challenging but solvable in 10 minutes. Do not provide a solution or code block. Keep it to one paragraph."
    
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        # Extract and clean up the string response
        q = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating dynamic coding question: {e}")
        # Adaptive coding question fallback
        q = (
            "Write an optimized Python function to determine if a given string is a palindrome."
            if is_tech
            else "Write a Python script to iterate through and print numbers from 1 to 10."
        )
        
    session["coding_q"] = q
    return render_template("coding.html", question=q)


@app.route("/submit_code", methods=["POST"])
def submit_code():
    code = request.form.get("code", "")
    reasoning_text = "No reasoning provided."

    # Transcribe audio if provided
    if "audio" in request.files:
        audio_file = request.files["audio"]
        if audio_file.filename != "":
            temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_logic.wav")
            audio_file.save(temp_path)
            try:
                reasoning_text = transcribe_audio(temp_path)
            except Exception as e:
                print("Audio Transcription Error:", e)

    q = session.get("coding_q", "Unknown Problem")

    prompt = f"""
    Task: {q}
    Candidate's Code Submission:
    ```python
    {code}
    ```
    Candidate's Spoken Logical Reasoning:
    "{reasoning_text}"

    You must evaluate this submission based on BOTH the code correctness and the logical reasoning explained.

    Format your response EXACTLY as a valid JSON object. Do not include markdown blocks or conversational text.
    {{
        "technical_score": <number 0-100 indicating code correctness and efficiency>,
        "communication_score": <number 0-100 indicating clarity of thought and logical reasoning>,
        "feedback": "<A concise technical critique of the execution>",
        "optimization": "<A short suggestion on alternative approaches or optimizations (time/space complexity)>"
    }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
        )

        raw_content = response.choices[0].message.content.strip()
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()

        eval_data = json.loads(raw_content)

        session["coding_score"] = eval_data.get("technical_score", 0)
        session["communication_score"] = eval_data.get("communication_score", 0)
        session["coding_feedback"] = eval_data.get("feedback", "No feedback.")
        session["coding_optimization"] = eval_data.get(
            "optimization", "No optimizations suggested."
        )

    except Exception as e:
        print(f"Error evaluating code: {e}")
        session["coding_score"] = 0
        session["communication_score"] = 0
        session["coding_feedback"] = "Error during AI evaluation."
        session["coding_optimization"] = "N/A"

    return jsonify({"redirect": url_for("dashboard")})


@app.route("/dashboard")
def dashboard():
    res = session.get("resume_final_pct", 0)
    cog = session.get("cognitive_score", 0)
    cod = session.get("coding_score", 0)
    com = session.get("communication_score", 0)
    cheat_attempts = session.get("cheating_attempts", 0)
    cod_feedback = session.get("coding_feedback", "No feedback available.")
    cod_opt = session.get("coding_optimization", "No optimizations given.")

    overall = (
        (res * WEIGHTS["resume"])
        + (cod * WEIGHTS["coding"])
        + (cog * WEIGHTS["cognitive"])
    )
    passed_cog = cog >= 80

    # Generate a brief summary of feedback
    feedback_summary = "Qualified" if passed_cog else "Failed Cognitive Phase"

    # Only save to DB if a resume score is present (meaning an interview actually happened)
    if res > 0 and session.get("session_saved") is None:
        save_session_to_db(
            round(res, 2),
            round(cog, 2),
            round(cod, 2),
            round(com, 2),
            round(overall, 2),
            cheat_attempts,
            feedback_summary,
        )
        session["session_saved"] = True  # Prevent duplicate saves on refresh

    # Fetch history
    history = []
    sim_history = []
    try:
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT date, resume_score, cognitive_score, coding_score, communication_score, overall_score, cheating_attempts, feedback_summary FROM interviews ORDER BY id DESC LIMIT 10"
        )
        history = cursor.fetchall()

        cursor.execute(
            "SELECT date, company, role, percentage, apti_score, dsa_score, hr_score, ai_feedback FROM company_simulations ORDER BY id DESC LIMIT 10"
        )
        sim_history = cursor.fetchall()

        conn.close()
    except Exception as e:
        print(f"Error fetching history: {e}")

    # Extract missing skills for Practice Arena
    missing_skills_set = set()
    analysis_data = session.get("analysis_data", None)
    
    if analysis_data:
        try:
            recs = get_vacancy_recommendations(analysis_data, top_n=5)
            for r in recs:
                for missing in r.get("missing", []):
                    missing_skills_set.add(missing.title())
        except Exception as e:
            print(f"Error fetching recommendations for missing skills: {e}")
            
    missing_skills = list(missing_skills_set)[:6]

    return render_template(
        "dashboard.html",
        resume=round(res, 2),
        cognitive=round(cog, 2),
        coding=round(cod, 2),
        communication=round(com, 2),
        overall=round(overall, 2),
        passed_cog=passed_cog,
        cheating_attempts=cheat_attempts,
        history=history,
        coding_feedback=cod_feedback,
        coding_optimization=cod_opt,
        analysis_data=analysis_data,
        missing_skills=missing_skills,
        sim_history=sim_history
    )


@app.route("/submit_resume_voice", methods=["POST"])
def submit_resume_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_audio.wav")
    audio_file.save(temp_path)

    # 🎤 Convert speech → text
    answer_text = transcribe_audio(temp_path)
    print(f"User Spoke: {answer_text}")

    q_list = session.get("resume_questions")
    idx = session.get("resume_idx", 0)

    if not q_list:
        return jsonify({"redirect": url_for("home")})

    # 🔥 CRITICAL FIX: boundary check BEFORE accessing list
    if idx >= len(q_list):
        return jsonify({"redirect": url_for("dashboard")})

    score, feedback, followup = evaluate_answer_ai(q_list[idx], answer_text)
    session["resume_score_total"] += score

    # Logic for Follow-up question
    if followup and score > 7 and "followup_active" not in session:
        session["followup_active"] = True
        return jsonify(
            {"question": followup, "feedback": feedback, "transcript": answer_text}
        )

    session.pop("followup_active", None)
    session["resume_idx"] += 1

    # End of round logic
    if session["resume_idx"] >= len(q_list):
        session["resume_final_pct"] = (
            session["resume_score_total"] / (len(q_list) * 10)
        ) * 100
        return jsonify({"redirect": url_for("dashboard")})

    # Standard next question
    return jsonify(
        {
            "question": q_list[session["resume_idx"]],
            "feedback": feedback,
            "transcript": answer_text,
        }
    )


@app.route("/voice-answer", methods=["POST"])
def voice_answer():
    audio_file = request.files["audio"]
    file_path = "temp_audio.wav"
    audio_file.save(file_path)
    text = transcribe_audio(file_path)
    return jsonify({"text": text})


@app.route("/log_cheat", methods=["POST"])
def log_cheat():
    if "cheating_attempts" in session:
        session["cheating_attempts"] += 1
    else:
        session["cheating_attempts"] = 1
    return jsonify({"status": "logged", "attempts": session["cheating_attempts"]})


# =========================
# DSA PRACTICE ARENA
# =========================


@app.route("/practice")
def practice():
    """Generates a dynamic DSA problem based on the candidate's resume skills."""
    skills = session.get("skills", "general programming, arrays, strings")
    difficulty = request.args.get("difficulty", "medium")

    prompt = f"""You are a DSA interviewer. Generate a single, well-defined coding problem.
Candidate skills: {skills}
Difficulty: {difficulty}
Rules:
- The problem must be original and solvable in Python.
- Include a clear problem statement, 2 example inputs/outputs, and constraints.
- Do NOT provide the solution or any hints about the approach.
- The problem should be relevant to the candidate's background if technical.

Format your response EXACTLY as a JSON object:
{{
    "title": "Problem Title",
    "difficulty": "{difficulty}",
    "statement": "Full problem description",
    "examples": [
        {{"input": "example input 1", "output": "expected output 1"}},
        {{"input": "example input 2", "output": "expected output 2"}}
    ],
    "constraints": "List of constraints"
}}
Return ONLY the JSON. No markdown, no extra text."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
        )
        raw = response.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        problem = json.loads(raw)
    except Exception as e:
        print(f"Practice question generation error: {e}")
        problem = {
            "title": "Two Sum",
            "difficulty": difficulty,
            "statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume each input has exactly one solution, and you may not use the same element twice.",
            "examples": [
                {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"},
                {"input": "nums = [3,2,4], target = 6", "output": "[1,2]"},
            ],
            "constraints": "2 <= nums.length <= 10^4, -10^9 <= nums[i] <= 10^9, Only one valid answer exists.",
        }

    session["practice_problem"] = problem
    session["practice_chat_history"] = []
    return render_template("practice.html", problem=problem, difficulty=difficulty)


@app.route("/practice_voice_chat", methods=["POST"])
def practice_voice_chat():
    """Socratic AI coach: responds to candidate voice/text without revealing the answer."""
    user_message = ""

    if "audio" in request.files and request.files["audio"].filename != "":
        audio_file = request.files["audio"]
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], "practice_voice.wav")
        audio_file.save(temp_path)
        try:
            user_message = transcribe_audio(temp_path)
        except Exception as e:
            return jsonify({"error": f"Transcription failed: {e}"}), 500
    else:
        user_message = request.form.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message received"}), 400

    problem = session.get("practice_problem", {})
    chat_history = session.get("practice_chat_history", [])

    system_prompt = f"""You are an expert DSA interview coach in a PRACTICE session.
The candidate is working on: "{problem.get('title', 'a coding problem')}"
Problem: {problem.get('statement', '')}

Your STRICT rules:
1. NEVER reveal the solution, working code, or the complete algorithm.
2. Guide using the Socratic method — ask questions that lead their thinking.
3. If they are on the right track, affirm and push deeper.
4. If they are stuck or wrong, give a gentle nudge without the answer.
5. Suggest whether they should reconsider time/space trade-offs.
6. Keep responses SHORT (2-4 sentences max) — this will be spoken aloud as voice.
7. Never write code. Pseudocode only if absolutely necessary, one line max.
8. If they ask for the direct answer, refuse: "I'm here to guide you, not give it away!"
9. Always end with a question to keep their thinking going."""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["coach"]})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.6, max_tokens=200
        )
        coach_reply = response.choices[0].message.content.strip()
    except Exception as e:
        coach_reply = (
            "Let me think with you! Can you walk me through what you're considering?"
        )

    chat_history.append({"user": user_message, "coach": coach_reply})
    session["practice_chat_history"] = chat_history

    return jsonify({"user_message": user_message, "coach_reply": coach_reply})


@app.route("/practice_analyze", methods=["POST"])
def practice_analyze():
    """Analyzes the candidate's code approach without revealing the optimal solution."""
    code = request.form.get("code", "").strip()
    problem = session.get("practice_problem", {})

    if not code:
        return (
            jsonify({"feedback": "No code submitted yet. Write something first!"}),
            400,
        )

    prompt = f"""You are a code approach reviewer. The candidate submitted code for: "{problem.get('title', 'a DSA problem')}"
Problem: {problem.get('statement', '')}

Candidate's Code:
```python
{code}
```

Evaluate ONLY the APPROACH — not nitpick syntax. Guide, don't judge harshly.

Respond in this EXACT JSON format:
{{
    "approach_name": "Name of the approach used (e.g. Brute Force, Hash Map, Two Pointers, Sliding Window, DP, etc.)",
    "time_complexity": "Big-O time complexity",
    "space_complexity": "Big-O space complexity",
    "approach_quality": "good|acceptable|can_improve",
    "what_they_did_well": "1-2 sentences on positives",
    "hint_to_improve": "A directional hint WITHOUT giving the answer",
    "better_approach_exists": true or false
}}
Return ONLY the JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        analysis = json.loads(raw)
    except Exception as e:
        print(f"Practice analyze error: {e}")
        analysis = {
            "approach_name": "Unable to detect",
            "time_complexity": "Unknown",
            "space_complexity": "Unknown",
            "approach_quality": "can_improve",
            "what_they_did_well": "You made an attempt — that's the first step!",
            "hint_to_improve": "Review whether your approach handles the constraints efficiently.",
            "better_approach_exists": True,
        }

    return jsonify(analysis)


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    """Saves user inquiry / feedback form submission to the database."""
    data = request.json
    name = data.get("name", "Anonymous")
    email = data.get("email", "")
    category = data.get("category", "General")
    message = data.get("message", "")
    rating = data.get("rating", 5)

    if not message:
        return jsonify({"status": "error", "message": "Message cannot be empty."}), 400

    try:
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO user_feedback (date, name, email, category, message, rating)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (current_date, name, email, category, message, rating),
        )
        conn.commit()
        conn.close()
        return jsonify(
            {
                "status": "success",
                "message": "Thank you for your feedback! We will get back to you shortly.",
            }
        )
    except Exception as e:
        print(f"Feedback DB Error: {e}")
        return jsonify({"status": "error", "message": "Failed to save feedback."}), 500


@app.route("/api/helpdesk", methods=["POST"])
def helpdesk_query():
    """AI-powered help desk that answers common user questions about the platform."""
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question."}), 400

    system_prompt = """You are a friendly and concise Help Desk assistant for an AI-powered Interview Simulation Platform called 'Inferno'.
    The platform has these rounds:
    1. Resume Round: 5 AI-generated questions based on resume skills. Voice or text answers accepted.
    2. Cognitive Round: 5 MCQs on Aptitude and Core Engineering. 80% pass threshold.
    3. Coding Round: One coding problem, submit code + voice reasoning. AI evaluates both.
    4. Dashboard: Shows all scores including Communication score (clarity of voice explanation in Coding Round).
    5. Practice Arena: DSA practice with Socratic AI coach and code analysis.

    Key facts:
    - Communication Score = clarity and logical reasoning of SPOKEN voice explanation during Coding Round (0-100).
    - Cheating detection: Tab switching and focus loss are logged.
    - All sessions are persisted in history with date, scores, violations, and result status.
    - Users can submit feedback via the Inquiry Form on the dashboard footer.

    Keep answers under 4 sentences. Be helpful, warm, and direct."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = (
            "I'm having trouble connecting right now. Please try again in a moment."
        )

    return jsonify({"answer": answer})


# =========================
# JOB RECOMMENDATION ENGINE
# =========================


def get_vacancy_recommendations(analysis_data, top_n=20, role_filter=""):
    try:
        from placement_engine import _db, compute_vacancy_match, _json_loads_safe

        conn = _db()
        rows = conn.execute("SELECT * FROM vacancies ORDER BY id DESC").fetchall()
        conn.close()
    except Exception as e:
        print("Error fetching vacancies:", e)
        rows = []

    verified = analysis_data.get("verified_skills", [])
    partial = analysis_data.get("partial_skills", [])
    predicted = analysis_data.get("predicted_role", "")

    results = []
    for r in rows:
        company = r["company_name"]
        role_name = r["job_role"]
        desc = r["description"] or f"Opportunity at {company}"
        req_skills = _json_loads_safe(r["required_skills"])
        gth_skills = _json_loads_safe(r["good_to_have"])

        if (
            role_filter
            and role_filter not in role_name.lower()
            and role_filter not in company.lower()
        ):
            continue

        match_res = compute_vacancy_match(verified, partial, req_skills, gth_skills)

        results.append(
            {
                "id": r["id"],
                "title": f"{role_name} @ {company}",
                "icon": "💼",
                "desc": desc,
                "min_cgpa": r["min_cgpa"],
                "ctc": r["ctc"],
                "score": match_res["score"],
                "exact_hits": match_res["exact_hits"],
                "partial_hits": match_res["partial_hits"],
                "missing": match_res["missing"],
                "is_predicted": (role_name.lower() in predicted.lower()),
            }
        )

    results.sort(key=lambda x: (x["is_predicted"], x["score"]), reverse=True)
    return results[:top_n]


@app.route("/job_recommendations")
def job_recommendations():
    """Renders the job recommendation page based on deep resume analysis stored in session."""
    analysis_data = session.get("analysis_data")
    if not analysis_data:
        return redirect(url_for("home"))

    # Also make sure the user passed a valid resume_text in session for Auto Apply
    recommendations = get_vacancy_recommendations(analysis_data, top_n=10)
    return render_template(
        "recommendations.html",
        analysis=analysis_data,
        recommendations=recommendations,
        # We pass dummy logic variables to handle the auto apply
        candidate_name=session.get("name", "User"),
        candidate_email=session.get("email", "info@example.com"),
        assessment_score=session.get(
            "resume_final_pct", session.get("cognitive_score", 50)
        ),
    )


@app.route("/api/job_recommendations", methods=["POST"])
def api_job_recommendations():
    """
    Live API endpoint: accepts a job role filter and returns recalculated rankings.
    """
    analysis_data = session.get("analysis_data", {})
    role_filter = request.json.get("role_filter", "").lower().strip()

    recommendations = get_vacancy_recommendations(
        analysis_data, top_n=20, role_filter=role_filter
    )
    return jsonify({"recommendations": recommendations})


@app.route("/mock_review")
def mock_review():
    return render_template("mock_review.html")

@app.route("/portfolio")
def portfolio():
    return render_template("modulo.html")

@app.route("/resume_maker")
def resume_maker():
    return render_template("resume_maker.html")

@app.route("/company_simulation")
def company_simulation():
    return render_template("company_simulation.html")

@app.route("/api/submit_company_simulation", methods=["POST"])
def submit_company_simulation():
    data = request.json
    name = session.get("name", "Student") # default or fetched from session
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    company = data.get("company", "Unknown")
    role = data.get("role", "Unknown")
    apti = data.get("apti_score", 0)
    dsa = data.get("dsa_score", 0)
    hr = data.get("hr_score", 0)
    total_pct = data.get("total_pct", 0)

    prompt = f"As an AI career coach from '{company}', analyze this candidate's performance for the '{role}' role. Their scores are: Aptitude: {apti}%, DSA: {dsa}%, HR: {hr}%. Provide a very brief (3-4 sentences) personalized analytical review of their performance and what skills they need for future upgradation. Keep it professional."

    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.6, max_tokens=150
        )
        ai_feedback = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        ai_feedback = "Your performance was recorded. Keep practicing to improve your skills for this role."

    try:
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO company_simulations (date, candidate_name, company, role, score, total, percentage, apti_score, dsa_score, hr_score, ai_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (current_date, name, company, role, 0, 0, total_pct, apti, dsa, hr, ai_feedback),
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "ai_feedback": ai_feedback})
    except Exception as e:
        print(f"Company simulation DB Error: {e}")
        return jsonify({"status": "error"}), 500



# =========================
# MOCK CODE REVIEW SIMULATOR
# =========================

@app.route("/review/generate", methods=["POST"])
def review_generate():
    """Generates intentionally flawed code for the candidate to review."""
    difficulty = request.json.get("difficulty", "easy")
    prompt = f"""You are a senior engineer creating a code review exercise.
Generate a Python function (20-40 lines) with INTENTIONAL security and performance flaws.
Difficulty: {difficulty}
Include at least 3 flaws: SQL injection, hardcoded passwords, missing validation, O(N^2) algorithm, weak hashing.
Return ONLY valid JSON:
{{{{
    "code": "the full python code",
    "vulnerabilities": ["list of vulnerability names"]
}}}}
No markdown. ONLY JSON."""
    try:
        response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8)
        raw = response.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        return jsonify(json.loads(raw))
    except Exception as e:
        print(f"Review generate error: {e}")
        fallback_code = (
            "import sqlite3\nimport hashlib\n\n"
            "DB_PASSWORD = \'admin123\'\n\n"
            "def get_user(username):\n"
            "    conn = sqlite3.connect(\'app.db\')\n"
            "    query = \"SELECT * FROM users WHERE name = \'\"+username+\"\'\"\n"
            "    result = conn.execute(query).fetchall()\n"
            "    conn.close()\n"
            "    return result\n\n"
            "def find_duplicates(items):\n"
            "    duplicates = []\n"
            "    for i in range(len(items)):\n"
            "        for j in range(i + 1, len(items)):\n"
            "            if items[i] == items[j]:\n"
            "                if items[i] not in duplicates:\n"
            "                    duplicates.append(items[i])\n"
            "    return duplicates\n\n"
            "def hash_password(password):\n"
            "    return hashlib.md5(password.encode()).hexdigest()\n"
        )
        return jsonify({
            "code": fallback_code,
            "vulnerabilities": ["SQL Injection", "Hardcoded password", "O(N^2) algorithm", "Weak MD5 hash", "No input validation"]
        })


@app.route("/review/evaluate", methods=["POST"])
def review_evaluate():
    """AI evaluates the candidate's code review feedback."""
    code = request.json.get("code", "")
    vulnerabilities = request.json.get("vulnerabilities", [])
    feedback = request.json.get("feedback", "")
    if not feedback:
        return jsonify({"score": 0, "evaluation": "No feedback provided."}), 400
    prompt = f"""Evaluate this code review 0-100.
Flawed code:
{code}

Hidden vulnerabilities: {json.dumps(vulnerabilities)}
Candidate review: "{feedback}"
Score: vulnerabilities found 40%, fix quality 30%, clarity 15%, security awareness 15%.
Return ONLY JSON: {{"score": <0-100>, "evaluation": "2-4 sentence assessment"}}"""
    try:
        response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        raw = response.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        return jsonify(json.loads(raw))
    except Exception as e:
        print(f"Review evaluate error: {e}")
        return jsonify({"score": 50, "evaluation": "Evaluation error. Please try again."})

# =========================
# MISSING PAGE ROUTES
# =========================

@app.route("/github-analyzer")
def github_analyzer_page():
    return render_template("github.html")

@app.route("/hr-login", methods=["GET", "POST"])
def hr_login_page():
    if request.method == "POST":
        email = request.form.get("email")
        if email == "hrtcsnagpur1@gmail.com":
            session["hr_authenticated"] = True
            return redirect("/placement/hr_dashboard")
        else:
            return "Unauthorized HR Email", 403
    return '''
    <html>
    <body style="background:#0f172a; color:white; font-family:sans-serif; display:flex; justify-content:center; align-items:center; height:100vh;">
      <form method="POST" style="background:#1e293b; padding:2.5rem; border-radius:1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); text-align:center;">
        <h2 style="margin-top:0;">🔒 HR Portal Login</h2>
        <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:1.5rem;">Access restricted to authorized HR personnel.</p>
        <input name="email" type="email" placeholder="Enter HR Email" required style="padding:0.75rem; margin-bottom:1rem; width:100%; border-radius:0.5rem; border:1px solid #334155; background:#0f172a; color:white; box-sizing:border-box;"><br>
        <button type="submit" style="padding:0.75rem 1.5rem; background:#3b82f6; border:none; border-radius:0.5rem; color:white; font-weight:bold; cursor:pointer; width:100%;">Authenticate</button>
      </form>
    </body>
    </html>
    '''


# =========================
# XP & GAMIFICATION ENGINE
# =========================

XP_AWARDS = {
    "company_simulation": 50,
    "coding_round": 40,
    "cognitive_test": 30,
    "resume_interview": 20,
    "roadmap_task": 10,
    "video_interview": 45,
    "mock_review": 35,
    "daily_login": 5,
}

LEVEL_THRESHOLDS = [0, 100, 250, 500, 850, 1300, 1900, 2650, 3600, 5000, 7000]

def get_level(xp):
    for i, threshold in enumerate(LEVEL_THRESHOLDS):
        if xp < threshold:
            return max(1, i)
    return len(LEVEL_THRESHOLDS)

def get_level_progress(xp):
    level = get_level(xp)
    if level >= len(LEVEL_THRESHOLDS):
        return 100
    prev = LEVEL_THRESHOLDS[level - 1]
    next_t = LEVEL_THRESHOLDS[level]
    return round(((xp - prev) / (next_t - prev)) * 100)

def update_xp(email, activity, score=0, username="User"):
    xp_earned = XP_AWARDS.get(activity, 10)
    bonus = int(score * 0.5) if score > 80 else 0
    xp_earned += bonus
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect("history.db")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM user_xp WHERE email=?", (email,)).fetchone()
        if row:
            new_xp = row["xp"] + xp_earned
            new_level = get_level(new_xp)
            streak = row["streak"]
            last_d = row["last_activity_date"] or ""
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            if last_d == today:
                pass  # same day, streak stays
            elif last_d == yesterday:
                streak += 1  # consecutive
            else:
                streak = 1  # reset
            new_best = max(row["best_score"], score)
            conn.execute("UPDATE user_xp SET xp=?, level=?, streak=?, last_activity_date=?, total_interviews=total_interviews+1, best_score=? WHERE email=?",
                        (new_xp, new_level, streak, today, new_best, email))
        else:
            conn.execute("INSERT INTO user_xp (email, username, xp, level, streak, last_activity_date, total_interviews, best_score) VALUES (?,?,?,?,?,?,?,?)",
                        (email, username, xp_earned, get_level(xp_earned), 1, today, 1, score))
        conn.execute("INSERT INTO xp_log (email, date, activity, xp_earned, score) VALUES (?,?,?,?,?)",
                    (email, now_str, activity, xp_earned, score))
        conn.commit()
        conn.close()
        return xp_earned
    except Exception as e:
        print(f"XP update error: {e}")
        return 0

@app.route("/leaderboard")
def leaderboard():
    target_role = request.args.get("role", "")
    try:
        conn = sqlite3.connect("history.db")
        conn.row_factory = sqlite3.Row
        if target_role:
            rows = conn.execute("SELECT username, xp, level, streak, total_interviews, best_score, target_role FROM user_xp WHERE target_role LIKE ? ORDER BY xp DESC LIMIT 50", (f"%{target_role}%",)).fetchall()
        else:
            rows = conn.execute("SELECT username, xp, level, streak, total_interviews, best_score, target_role FROM user_xp ORDER BY xp DESC LIMIT 50").fetchall()
        conn.close()
        leaderboard_data = [dict(r) for r in rows]
    except Exception as e:
        print(f"Leaderboard error: {e}")
        leaderboard_data = []
    return render_template("leaderboard.html", leaderboard=leaderboard_data, role_filter=target_role)

@app.route("/api/award_xp", methods=["POST"])
def award_xp():
    data = request.json
    email = data.get("email", "anonymous@user.com")
    activity = data.get("activity", "daily_login")
    score = data.get("score", 0)
    username = data.get("username", email.split("@")[0])
    xp = update_xp(email, activity, score, username)
    return jsonify({"xp_earned": xp, "status": "ok"})

@app.route("/api/get_user_stats", methods=["POST"])
def get_user_stats():
    email = request.json.get("email", "")
    try:
        conn = sqlite3.connect("history.db")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM user_xp WHERE email=?", (email,)).fetchone()
        recent = conn.execute("SELECT activity, xp_earned, score, date FROM xp_log WHERE email=? ORDER BY id DESC LIMIT 5", (email,)).fetchall()
        conn.close()
        if row:
            d = dict(row)
            d["level_progress"] = get_level_progress(d["xp"])
            d["xp_for_next"] = LEVEL_THRESHOLDS[min(d["level"], len(LEVEL_THRESHOLDS)-1)]
            d["recent"] = [dict(r) for r in recent]
            return jsonify(d)
        return jsonify({"xp": 0, "level": 1, "streak": 0, "total_interviews": 0, "best_score": 0, "level_progress": 0})
    except Exception as e:
        return jsonify({"xp": 0, "level": 1, "streak": 0}), 500


# =========================
# NATIVE ROADMAP GENERATOR
# =========================

@app.route("/roadmap")
def roadmap():
    analysis_data = session.get("analysis_data")
    if not analysis_data:
        return redirect(url_for("home"))
    role = analysis_data.get("predicted_role", "Software Engineer")
    missing = analysis_data.get("partial_skills", []) + ["system design", "communication"]
    return render_template("roadmap.html", role=role, analysis=analysis_data)

@app.route("/api/generate_roadmap", methods=["POST"])
def generate_roadmap():
    role = request.json.get("role", "Software Engineer")
    missing_skills = request.json.get("missing_skills", [])
    timeframe = request.json.get("timeframe", "8")  # weeks
    prompt = f"""You are an expert career coach. Generate a highly specific, actionable {timeframe}-week learning roadmap for someone targeting the role of '{role}'.
Their missing/weak skills are: {', '.join(missing_skills[:8]) if missing_skills else 'general programming fundamentals'}.

Create a week-by-week roadmap. Each week should have:
- A clear theme/focus area
- Exactly 3-4 specific actionable tasks (e.g. "Build a REST API with FastAPI", not "learn Python")
- One milestone project or goal
- Estimated daily hours (1-3 hours)

Return ONLY valid JSON in this EXACT format:
{{
  "role": "{role}",
  "total_weeks": {timeframe},
  "weeks": [
    {{
      "week": 1,
      "theme": "Foundation & Setup",
      "daily_hours": 2,
      "tasks": [
        "Task description 1",
        "Task description 2",
        "Task description 3"
      ],
      "milestone": "Milestone project/goal for this week",
      "resources": ["Resource 1", "Resource 2"]
    }}
  ]
}}
Return ONLY the JSON. No markdown. No explanation."""
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=2000
        )
        raw = response.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        return jsonify(data)
    except Exception as e:
        print(f"Roadmap generation error: {e}")
        return jsonify({"error": "Failed to generate roadmap"}), 500

@app.route("/api/toggle_roadmap_task", methods=["POST"])
def toggle_roadmap_task():
    data = request.json
    email = data.get("email", "anonymous")
    role = data.get("role", "")
    week = data.get("week", 0)
    task_idx = data.get("task_idx", 0)
    completed = data.get("completed", 1)
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect("history.db")
        existing = conn.execute("SELECT id FROM roadmap_progress WHERE email=? AND role=? AND week=? AND task_idx=?",
                               (email, role, week, task_idx)).fetchone()
        if existing:
            conn.execute("UPDATE roadmap_progress SET completed=?, date=? WHERE id=?", (completed, today, existing[0]))
        else:
            conn.execute("INSERT INTO roadmap_progress (email, role, week, task_idx, completed, date) VALUES (?,?,?,?,?,?)",
                        (email, role, week, task_idx, completed, today))
        conn.commit()
        conn.close()
        if completed:
            update_xp(email, "roadmap_task")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error"}), 500


# =========================
# VIDEO / BEHAVIORAL INTERVIEW
# =========================

BEHAVIORAL_QUESTIONS = [
    "Tell me about a time you overcame a major technical challenge. Walk me through your thought process.",
    "Describe a situation where you had to learn a new technology quickly. How did you approach it?",
    "Tell me about a project you're most proud of. What was your specific contribution and what impact did it have?",
    "How do you handle disagreements with teammates about technical decisions?",
    "Describe a time you had to debug a complex issue under pressure. What was the process?",
    "Tell me about a time you failed on a project. What did you learn?",
    "How do you prioritize tasks when everything seems urgent?",
    "Describe a situation where you had to explain a complex technical concept to a non-technical stakeholder.",
    "Tell me about a time you went above and beyond your assigned responsibilities.",
    "How do you stay updated with the latest technology trends in your field?"
]

@app.route("/video_interview")
def video_interview():
    analysis_data = session.get("analysis_data", {})
    role = analysis_data.get("predicted_role", "Software Engineer") if analysis_data else "Software Engineer"
    # Pick 3 random questions
    import random as rnd
    questions = rnd.sample(BEHAVIORAL_QUESTIONS, min(3, len(BEHAVIORAL_QUESTIONS)))
    return render_template("video_interview.html", questions=questions, role=role)

@app.route("/api/evaluate_video_answer", methods=["POST"])
def evaluate_video_answer():
    question = request.json.get("question", "")
    transcript = request.json.get("transcript", "")
    role = request.json.get("role", "Software Engineer")
    if not transcript or len(transcript.strip()) < 20:
        return jsonify({"score": 0, "feedback": "Answer too short or unclear. Please speak clearly.", "confidence": 0}), 400
    prompt = f"""You are a senior {role} interviewer conducting a behavioral interview.
Question asked: "{question}"
Candidate's spoken answer (transcript): "{transcript}"

Evaluate this behavioral answer on:
1. STAR Method usage (40%) - did they give Situation, Task, Action, Result?
2. Relevance & Specificity (30%) - is it specific or vague?
3. Communication Clarity (20%) - clear, structured, confident?
4. Impact Focus (10%) - did they mention outcomes/results?

Return ONLY valid JSON:
{{
  "score": <0-100>,
  "confidence_score": <0-100 estimate of speaker confidence based on answer quality>,
  "strengths": "1-2 specific things they did well",
  "improvements": "1-2 specific things to improve",
  "better_example": "A brief example of how to improve their answer",
  "star_rating": {{ "situation": <0-10>, "task": <0-10>, "action": <0-10>, "result": <0-10> }}
}}"""
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        # Save to DB
        email = request.json.get("email", "anonymous")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn = sqlite3.connect("history.db")
            conn.execute("INSERT INTO video_interviews (email, date, question, transcript, score, feedback, confidence_score) VALUES (?,?,?,?,?,?,?)",
                        (email, now_str, question, transcript, data.get("score", 0), data.get("strengths", ""), data.get("confidence_score", 0)))
            conn.commit()
            conn.close()
        except Exception as db_e:
            print(f"Video interview DB error: {db_e}")
        return jsonify(data)
    except Exception as e:
        print(f"Video interview eval error: {e}")
        return jsonify({"score": 60, "feedback": "Good attempt! Try to use the STAR method more explicitly.", "confidence": 65})


# =========================
# ENHANCED PR REVIEW (File Upload)
# =========================

@app.route("/review/upload_analyze", methods=["POST"])
def review_upload_analyze():
    """Accept a user's own code file and perform a senior-level AI PR review with line-by-line feedback."""
    code = ""
    filename = "uploaded_file.py"
    if "file" in request.files:
        f = request.files["file"]
        filename = secure_filename(f.filename)
        content = f.read().decode("utf-8", errors="ignore")
        code = content[:6000]  # limit
    elif request.form.get("code"):
        code = request.form.get("code")[:6000]

    if not code.strip():
        return jsonify({"error": "No code provided"}), 400

    prompt = f"""You are a Staff Software Engineer conducting a thorough Pull Request code review on '{filename}'.
Analyze this code like a senior engineer reviewing a real PR. Be specific, professional, and constructive.

Code to review:
```
{code}
```

Return ONLY valid JSON:
{{
  "overall_score": <0-100>,
  "verdict": "Approve | Request Changes | Needs Major Revision",
  "summary": "2-3 sentence overall assessment",
  "line_comments": [
    {{
      "line_range": "Lines 5-8",
      "severity": "critical|warning|suggestion",
      "issue": "Brief issue description",
      "fix": "Specific code fix or approach"
    }}
  ],
  "security_issues": ["issue1", "issue2"],
  "performance_issues": ["issue1"],
  "best_practices": ["suggestion1", "suggestion2"],
  "positive_aspects": ["what they did well 1", "what they did well 2"]
}}
Provide at least 3-5 line_comments. Return ONLY JSON, no markdown."""
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1500
        )
        raw = response.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        return jsonify(json.loads(raw))
    except Exception as e:
        print(f"PR upload analyze error: {e}")
        return jsonify({"error": "Analysis failed, please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)
