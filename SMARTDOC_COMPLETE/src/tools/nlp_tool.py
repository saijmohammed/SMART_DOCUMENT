"""
src/tools/nlp_tool.py — Outil d'explication de document (Tool #4).

Produit une explication simple, une liste de termes importants,
des questions d'étude, et le niveau de lecture estimé.

Schéma d'entrée  : { "text": str, "category": str }
Schéma de sortie : {
    "status"             : "success" | "error",
    "simple_explanation" : str,
    "important_terms"    : list[str],
    "study_questions"    : list[str],
    "reading_level"      : str,
    "mode"               : "gemini" | "local",
}
"""
import re
from collections import Counter

from src.tools.logger_tool  import log_action
from src.utils.config       import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.text_cleaner import truncate

_STOP = {
    "document", "cette", "faire", "avoir", "about", "which", "there", "their",
    "using", "comme", "pour", "avec", "dans", "sont", "plus", "aussi",
    "that", "will", "would", "could", "should", "shall", "your", "their",
    "aussi", "bien", "même", "très", "tout", "tous", "donc", "mais",
}


def _local_explain(text: str, category: str) -> dict:
    """Explication locale basée sur l'analyse de fréquences de mots."""
    words   = re.findall(r"[A-Za-zÀ-ÿ]{5,}", text.lower())
    freq    = Counter(w for w in words if w not in _STOP)
    terms   = [w for w, _ in freq.most_common(12)]
    n_words = len(text.split())

    if n_words < 300:
        level = "Simple"
    elif n_words < 1_500:
        level = "Intermédiaire"
    else:
        level = "Avancé"

    explanation = (
        f"Ce document appartient à la catégorie **{category or 'Non précisée'}**. "
        f"Il contient environ {n_words:,} mots (niveau de lecture : {level}). "
        "Le contenu a été automatiquement extrait et analysé. "
        "Les sujets principaux ont été identifiés à partir des termes les plus représentatifs. "
        "Consultez le résumé détaillé pour une synthèse structurée du contenu."
    )
    questions = [
        "Quel est l'objectif principal de ce document ?",
        "Quelles sont les sections ou idées les plus importantes ?",
        "Quelles conclusions peut-on tirer du contenu ?",
        "Quels concepts devrez-vous expliquer lors d'une présentation ?",
        "Y a-t-il des limites ou des questions ouvertes mentionnées dans le document ?",
    ]
    return {
        "simple_explanation": explanation,
        "important_terms"   : terms,
        "study_questions"   : questions,
        "reading_level"     : level,
    }


def _gemini_explain(text: str, category: str) -> dict:
    """Explication via l'API Gemini."""
    from google import genai  # type: ignore
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""You are an academic tutor helping a student understand a document.
Category: {category}

Document excerpt:
{truncate(text, 4_000)}

Respond in the document's language (French if the document is in French).
Provide strictly in this format:

EXPLANATION: (3-4 sentences in plain, simple language accessible to a non-expert)
TERMS: (comma-separated list of 10 key technical terms found in the text)
QUESTIONS: (5 study questions numbered 1-5)
LEVEL: (Simple / Intermédiaire / Avancé)
"""
    raw = client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()

    explanation, terms_raw, level = "", "", "Intermédiaire"
    questions: list[str] = []

    for line in raw.split("\n"):
        if line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("TERMS:"):
            terms_raw = line.replace("TERMS:", "").strip()
        elif line.startswith("LEVEL:"):
            level = line.replace("LEVEL:", "").strip()

    if "QUESTIONS:" in raw:
        q_block   = raw.split("QUESTIONS:")[1].split("LEVEL:")[0]
        questions = [
            re.sub(r"^\d+\.\s*", "", l).strip()
            for l in q_block.strip().split("\n") if l.strip()
        ]

    terms = [t.strip() for t in terms_raw.split(",") if t.strip()]

    return {
        "simple_explanation": explanation or "Voir le résumé ci-dessus.",
        "important_terms"   : terms[:12],
        "study_questions"   : questions[:5],
        "reading_level"     : level,
    }


def explain_text(text: str, category: str = "") -> dict:
    """
    Explique le contenu du document.
    Utilise Gemini si GEMINI_API_KEY est défini, sinon analyse locale.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Texte d'entrée vide.")

        if GEMINI_API_KEY:
            try:
                data = _gemini_explain(text, category)
                mode = "gemini"
            except Exception as gem_err:
                log_action("ExplainerAgent", "gemini_error", "warning",
                           {"error": str(gem_err)})
                data = _local_explain(text, category)
                mode = "local"
        else:
            data = _local_explain(text, category)
            mode = "local"

        log_action("ExplainerAgent", "explain_text", "success", {
            "mode" : mode,
            "terms": data["important_terms"][:5],
            "level": data["reading_level"],
        })
        return {"status": "success", "mode": mode, **data}

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log_action("ExplainerAgent", "explain_text", "error", {"error": err})
        return {"status": "error", "error": err}
