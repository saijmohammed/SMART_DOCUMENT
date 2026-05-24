"""
src/tools/summarizer_tool.py — Outil de résumé (Tool #3).

Stratégie (par ordre de priorité) :
  1. Gemini API  — si GEMINI_API_KEY est défini
  2. Résumé extractif TF-IDF amélioré — fallback Python pur

Schéma d'entrée  : { "text": str, "max_sentences": int }
Schéma de sortie : {
    "status"     : "success" | "error",
    "summary"    : str,
    "key_points" : list[str],
    "mode"       : "gemini" | "extractive",
    "title"      : str,
}
"""
import re
from collections import Counter

from src.tools.logger_tool  import log_action
from src.utils.config       import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.text_cleaner import chunk_text, truncate

_STOP = {
    "dans", "avec", "pour", "that", "this", "from", "vous", "nous",
    "they", "have", "plus", "comme", "des", "les", "une", "the",
    "and", "are", "was", "been", "will", "our", "your", "their",
    "which", "also", "mais", "donc", "puis", "après", "avant",
    "entre", "tout", "tous", "bien", "même", "très", "aussi",
    "est", "sont", "être", "avoir", "fait", "peut", "doit",
    "this", "that", "these", "those", "such", "some", "each",
    "pour", "dans", "avec", "sans", "sous", "vers", "entre",
}

# Patterns parasites issus de l'extraction PDF
_NOISE_PATTERNS = [
    r'Page\s+\d+\s+of\s+\d+',
    r'^\s*\d+\s*$',                       # Numéros de page seuls
    r'UIR\s*\|\s*\d{4}',                  # En-têtes UIR
    r'S\d+\s*—\s*Integrated Project',     # En-têtes de cours répétés
    r'AI\s*&\s*Big\s*Data\s*Program',     # Sous-titres répétés
    r'(?:^|\n)\s*\d+\s*(?:\n|$)',         # Lignes avec juste un numéro
]


def _clean_extracted_text(text: str) -> str:
    """Nettoie le texte extrait du PDF : retire les parasites, normalise."""
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.MULTILINE | re.IGNORECASE)
    # Retire les lignes très courtes (<15 chars) qui sont souvent des artefacts
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 14 or not l.strip()]
    text = '\n'.join(lines)
    # Normalise les espaces et sauts de ligne
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_title(text: str) -> str:
    """Tente d'extraire un titre significatif du document."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:10]:
        if 20 < len(line) < 120 and not re.search(r'\d{4}[-–]\d{4}', line):
            return line
    return ""


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    raw  = re.split(r'(?<=[.!?])\s+', text)
    # Filtre les phrases trop courtes ou parasites
    return [s.strip() for s in raw
            if len(s.strip()) > 30 and not re.match(r'^\d+\.?\s*$', s.strip())]


def _extractive_summary(text: str, max_sentences: int = 8) -> dict:
    """Résumé extractif amélioré : nettoie le texte avant de sélectionner les phrases."""
    clean = _clean_extracted_text(text)
    title = _extract_title(clean)

    sentences = _split_sentences(clean)
    if not sentences:
        sentences = _split_sentences(text)

    if not sentences:
        return {"summary": text[:600], "key_points": [], "title": title}

    if len(sentences) <= max_sentences:
        summary = " ".join(sentences)
    else:
        words  = re.findall(r'[A-Za-zÀ-ÿ]{4,}', clean.lower())
        freq   = Counter(w for w in words if w not in _STOP)
        scored = []
        for idx, sent in enumerate(sentences):
            sw    = re.findall(r'[A-Za-zÀ-ÿ]{4,}', sent.lower())
            score = sum(freq.get(w, 0) for w in sw) / max(1, len(sw))
            score += 0.6 / (idx + 1)   # préférence pour le début
            # Bonus si la phrase contient des chiffres ou des mots de structure
            if re.search(r'\b(objectif|goal|purpose|result|conclusion|require|must|shall)\b', sent.lower()):
                score += 2.0
            scored.append((score, idx, sent))
        top     = sorted(scored, reverse=True)[:max_sentences]
        top     = sorted(top, key=lambda x: x[1])
        summary = " ".join(s for _, _, s in top)

    # Points clés : phrases les plus informatives de chaque bloc
    key_points: list[str] = []
    for chunk in chunk_text(clean, max_chars=2_500)[:5]:
        sents = _split_sentences(chunk)
        for sent in sents[:4]:
            # Préfère les phrases qui commencent par un verbe d'action ou contiennent des chiffres
            if len(sent) > 40 and sent not in key_points:
                key_points.append(sent.strip())
            if len(key_points) >= 6:
                break
        if len(key_points) >= 6:
            break

    return {"summary": summary, "key_points": key_points[:6], "title": title}


def _gemini_summarize(text: str) -> dict:
    """Résumé via l'API Gemini — sortie structurée et lisible."""
    from google import genai  # type: ignore
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""You are a professional document analyst creating a high-quality, structured summary.

Given the document text below, produce:
1. TITLE: A concise, meaningful title for the document (if not obvious, infer it).
2. SUMMARY: A rich, fluent, well-structured summary in 5-7 clear sentences. 
   Write in proper paragraphs. Avoid copying raw text — paraphrase and synthesize.
   Cover the main purpose, key content, and conclusions.
3. KEY_POINTS: Exactly 6 key points as a numbered list. Each point should:
   - Start with a bold keyword or concept
   - Be a complete, self-contained sentence
   - Add value beyond the summary

Respond strictly in the document's language (French if document is in French).
Format:
TITLE:
<title>

SUMMARY:
<summary paragraphs>

KEY_POINTS:
1. ...
2. ...
3. ...
4. ...
5. ...
6. ...

Document:
{truncate(text, 6_000)}
"""
    raw = client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()

    title      = ""
    summary    = ""
    key_points = []

    if "TITLE:" in raw:
        after_title = raw.split("TITLE:", 1)[1]
        title       = after_title.split("\n")[1].strip() if "\n" in after_title else after_title[:120].strip()

    if "SUMMARY:" in raw and "KEY_POINTS:" in raw:
        parts   = raw.split("KEY_POINTS:", 1)
        summary = parts[0].split("SUMMARY:", 1)[-1].strip()
        for line in parts[1].strip().split("\n"):
            line = re.sub(r"^\d+\.\s*", "", line).strip()
            if line:
                key_points.append(line)
    elif "SUMMARY:" in raw:
        summary = raw.split("SUMMARY:", 1)[-1].strip()[:1_200]

    return {"summary": summary, "key_points": key_points[:6], "title": title}


def summarize_text(text: str, max_sentences: int = 8) -> dict:
    """
    Résume le texte.
    Utilise Gemini si GEMINI_API_KEY est défini, sinon résumé extractif amélioré.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Texte d'entrée vide.")

        if GEMINI_API_KEY:
            try:
                data = _gemini_summarize(text)
                mode = "gemini"
            except Exception as gem_err:
                log_action("SummarizerAgent", "gemini_error", "warning",
                           {"error": str(gem_err)})
                data = _extractive_summary(text, max_sentences)
                mode = "extractive"
        else:
            data = _extractive_summary(text, max_sentences)
            mode = "extractive"

        log_action("SummarizerAgent", "summarize_text", "success", {
            "mode"         : mode,
            "summary_chars": len(data["summary"]),
            "key_points"   : len(data["key_points"]),
        })
        return {"status": "success", "mode": mode, **data}

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log_action("SummarizerAgent", "summarize_text", "error", {"error": err})
        return {"status": "error", "error": err}
