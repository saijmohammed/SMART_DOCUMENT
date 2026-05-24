"""
src/utils/config.py
Centralisation de toute la configuration SmartDoc AI.
Toutes les constantes, chemins et variables d'environnement passent ici.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Racine du projet ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# ── Méta-application ──────────────────────────────────────────────────────────
APP_NAME    = os.getenv("APP_NAME", "SmartDoc AI")
APP_VERSION = "4.0.0"

# ── Chemins ───────────────────────────────────────────────────────────────────
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR    = BASE_DIR / "data"
LOGS_DIR    = BASE_DIR / "logs"
LOG_PATH    = LOGS_DIR / "actions.json"

for _p in [MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, DATA_DIR, LOGS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Clés API ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Hyper-paramètres modèle PyTorch ──────────────────────────────────────────
TFIDF_MAX_FEATURES = 4_000
TRAIN_EPOCHS       = 25
TRAIN_LR           = 1e-3
TRAIN_BATCH_SIZE   = 32
RANDOM_STATE       = 42

# ── Catégories de documents ───────────────────────────────────────────────────
DOC_CATEGORIES = [
    "Course / Academic",
    "Research Paper",
    "Report",
    "Business / Invoice",
    "Administrative",
    "Legal / Contract",
    "Medical / Health",
    "Technical / Engineering",
]
