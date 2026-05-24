"""
src/tools/chat_tool.py — Outil de chat contextuel (Tool #7).

Stratégie :
  1. Groq API    — llama-3.3-70b, rapide et gratuit
  2. Gemini API  — fallback si Groq absent
  3. Local smart — analyse structurée par type de question
"""
import re
from collections import Counter

from src.tools.logger_tool  import log_action
from src.utils.config       import (
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
)
from src.utils.text_cleaner import truncate

_STOP = {
    "le","la","les","un","une","des","du","de","et","en","est","à","au","aux",
    "que","qui","quoi","dans","sur","avec","pour","par","plus","comme","mais",
    "the","a","an","of","in","is","are","was","were","to","and","or","for",
    "this","that","with","from","have","has","not","be","it","they","we","you",
    "what","which","how","when","where","why","can","do","does","did","will",
    "quel","quelle","quels","quelles","sont","ses","son","leur","leurs",
    "vous","nous","ils","elles","je","tu","il","elle","me","te","se","ce","cet",
    "cette","ces","mon","ton","votre","notre","vos","nos","une","être",
}


# =============================================================================
# Prompt commun (Groq + Gemini)
# =============================================================================

def _build_prompt(question: str, document_text: str, meta: dict, history: list[dict]) -> str:
    doc_context = (
        f"Document : {meta.get('title','Non précisé')}\n"
        f"Catégorie : {meta.get('category','Non précisée')}\n"
        f"Pages : {meta.get('page_count','?')} | Mots : {meta.get('word_count','?')}\n\n"
        f"Résumé :\n{meta.get('summary','Non disponible')}\n\n"
        f"Texte (extrait) :\n{truncate(document_text, 6_000)}"
    )
    history_text = ""
    for msg in history[-8:]:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        history_text += f"{role} : {msg['content']}\n"

    return f"""Tu es un assistant expert en analyse documentaire. Réponds UNIQUEMENT à partir du document fourni.

RÈGLES DE FORMATAGE OBLIGATOIRES :
- Utilise **texte** pour mettre en gras les termes importants
- Utilise des titres avec ## pour chaque section principale
- Utilise des listes avec - pour chaque point (un point par ligne)
- Sépare chaque section par une ligne vide
- Chaque phrase importante = une ligne séparée
- Sois clair, structuré, lisible
- Réponds dans la langue de la question

=== DOCUMENT ===
{doc_context}

=== HISTORIQUE ===
{history_text if history_text else "(Début de conversation)"}

=== QUESTION ===
{question}

=== RÉPONSE (formatée avec ## titres et - listes) ==="""


# =============================================================================
# Groq
# =============================================================================

def _groq_answer(question: str, document_text: str, meta: dict, history: list[dict]) -> str:
    from groq import Groq  # type: ignore
    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_prompt(question, document_text, meta, history)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


# =============================================================================
# Gemini
# =============================================================================

def _gemini_answer(question: str, document_text: str, meta: dict, history: list[dict]) -> str:
    from google import genai  # type: ignore
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(question, document_text, meta, history)
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()


# =============================================================================
# Local intelligent — analyse structurée par type de question
# =============================================================================

def _detect_question_type(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["structure","organisé","plan","sections","parties","chapitres","composé"]):
        return "structure"
    if any(w in q for w in ["conclusion","conclure","résultat","bilan","findings","en fin","finalement"]):
        return "conclusion"
    if any(w in q for w in ["outil","technologie","logiciel","framework","langage","librairie","tool","stack","tech","plateforme","système"]):
        return "tools"
    if any(w in q for w in ["objectif","but","goal","purpose","vise","cherche","sujet","propos","concerne","traite"]):
        return "objective"
    if any(w in q for w in ["auteur","author","qui a écrit","rédigé","written by","professor","prof"]):
        return "author"
    if any(w in q for w in ["méthode","approche","comment","how","processus","étapes","démarche","procédure"]):
        return "method"
    if any(w in q for w in ["exigence","require","obligation","must","doit","critère","évaluation","grading","note"]):
        return "requirements"
    return "general"


def _split_paragraphs(text: str) -> list[str]:
    paras = re.split(r'\n{2,}', text)
    return [p.strip() for p in paras if len(p.strip()) > 50]


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+|\n', text)
    return [s.strip() for s in raw if len(s.strip()) > 40]


def _score_text(text: str, q_words: set) -> float:
    words = set(re.findall(r'[A-Za-zÀ-ÿ]{3,}', text.lower()))
    return len(words & q_words) / max(1, len(q_words))


def _extract_tech_terms(text: str) -> list[str]:
    known = ["CrewAI","LangGraph","LangChain","Ollama","OpenAI","Gemini","Python","FastAPI",
             "Flask","Django","React","Node","Docker","Kubernetes","MongoDB","PostgreSQL",
             "MySQL","Redis","TensorFlow","PyTorch","Scikit","Pandas","NumPy","Streamlit",
             "HuggingFace","Transformers","BERT","GPT","RAG","Vector","Pinecone","Chroma",
             "Weaviate","Faiss","AWS","Azure","GCP","GitHub","GitLab","CI/CD","REST","API",
             "JSON","XML","HTML","CSS","JavaScript","TypeScript","Java","C++","Go","Rust",
             "SQL","NoSQL","Spark","Hadoop","Airflow","MLflow","Jupyter","Colab"]
    found = []
    for term in known:
        if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            found.append(term)
    # Aussi chercher des mots en CamelCase ou MAJUSCULES inconnus
    caps = re.findall(r'\b[A-Z][a-z]+[A-Z]\w*\b|\b[A-Z]{2,}\b', text)
    extra = [c for c in caps if c not in ("The","This","It","In","Of","For","And","To","A","An","I") and len(c) > 2]
    combined = list(dict.fromkeys(found + extra))
    return combined[:20]


def _find_sections(text: str) -> list[str]:
    lines = text.split('\n')
    sections = []
    for line in lines:
        line = line.strip()
        # Titre : ligne courte, pas de ponctuation finale, capitalisée ou numérotée
        if (10 < len(line) < 100
                and not line.endswith('.')
                and (line[0].isupper() or re.match(r'^\d+[\.\)]\s', line))):
            sections.append(line)
    return sections[:15]


def _local_answer(question: str, document_text: str, meta: dict) -> dict:
    q_words  = set(re.findall(r'[A-Za-zÀ-ÿ]{3,}', question.lower())) - _STOP
    qtype    = _detect_question_type(question)
    title    = (meta.get("title") or "le document").strip()
    summary  = (meta.get("summary") or "").strip()
    category = (meta.get("category") or "").strip()
    pages    = meta.get("page_count", "?")
    words_n  = meta.get("word_count", "?")

    paragraphs = _split_paragraphs(document_text)
    sentences  = _split_sentences(document_text)

    # ── Structure ─────────────────────────────────────────────────────────────
    if qtype == "structure":
        sections = _find_sections(document_text)
        parts = []
        if sections:
            parts.append("**Structure du document :**\n")
            for i, s in enumerate(sections, 1):
                parts.append(f"{i}. {s}")
        else:
            # Fallback : découpage en blocs
            parts.append("**Organisation du document :**\n")
            for i, p in enumerate(paragraphs[:6], 1):
                preview = p[:120].rstrip() + ("…" if len(p) > 120 else "")
                parts.append(f"**Bloc {i} :** {preview}")
        if summary:
            parts.append(f"\n**Aperçu général :** {summary[:300]}…")
        return {"answer": "\n".join(parts), "sources": [], "mode": "local"}

    # ── Conclusion ────────────────────────────────────────────────────────────
    if qtype == "conclusion":
        conclusion_sents = [s for s in sentences
                            if any(w in s.lower() for w in
                                   ["conclusion","conclure","résultat","bilan","therefore",
                                    "finally","en conclusion","en résumé","objectif atteint",
                                    "recommand","perspect"])]
        # + dernières phrases du doc
        last_sents = sentences[-5:] if len(sentences) >= 5 else sentences
        combined   = list(dict.fromkeys(conclusion_sents + last_sents))[:5]
        if combined:
            answer = "**Conclusions et résultats du document :**\n\n"
            answer += "\n\n".join(f"• {s}" for s in combined)
        elif summary:
            answer = f"**Résumé conclusif :**\n\n{summary}"
        else:
            answer = "Aucune conclusion explicite détectée dans le document."
        return {"answer": answer, "sources": combined, "mode": "local"}

    # ── Outils / Technologies ─────────────────────────────────────────────────
    if qtype == "tools":
        techs = _extract_tech_terms(document_text)
        if techs:
            answer = f"**Technologies et outils mentionnés dans « {title} » :**\n\n"
            answer += "  ·  ".join(f"**{t}**" for t in techs)
            # Ajouter le contexte pour chaque outil clé
            answer += "\n\n**Contexte d'utilisation :**\n"
            for tech in techs[:5]:
                for sent in sentences:
                    if re.search(r'\b' + re.escape(tech) + r'\b', sent, re.IGNORECASE):
                        answer += f"\n• **{tech}** — {sent[:200]}"
                        break
        else:
            scored = sorted([(s, _score_text(s, q_words)) for s in sentences],
                            key=lambda x: x[1], reverse=True)
            top = [s for s, sc in scored if sc > 0][:4]
            answer = "**Éléments techniques mentionnés :**\n\n" + "\n\n".join(f"• {s}" for s in top) if top else "Aucun outil ou technologie explicitement mentionné."
        return {"answer": answer, "sources": [], "mode": "local"}

    # ── Objectif ──────────────────────────────────────────────────────────────
    if qtype == "objective":
        if summary:
            meta_line = f"**{title}**  ·  {category}  ·  {pages} pages  ·  {words_n} mots\n\n"
            return {"answer": meta_line + "**Objectif :**\n\n" + summary, "sources": [], "mode": "local"}
        obj_sents = [s for s in sentences
                     if any(w in s.lower() for w in ["objectif","but","goal","purpose","vise","aim","intend"])]
        if obj_sents:
            return {"answer": "**Objectif du document :**\n\n" + "\n\n".join(f"• {s}" for s in obj_sents[:3]),
                    "sources": obj_sents[:3], "mode": "local"}

    # ── Exigences / Évaluation ────────────────────────────────────────────────
    if qtype == "requirements":
        req_sents = [s for s in sentences
                     if any(w in s.lower() for w in
                            ["must","shall","doit","require","exige","obligatoire","mandatory",
                             "evaluation","grading","note","critère","deliverable","livrable"])]
        if req_sents:
            answer = "**Exigences et critères d'évaluation :**\n\n"
            answer += "\n\n".join(f"• {s}" for s in req_sents[:6])
            return {"answer": answer, "sources": req_sents[:6], "mode": "local"}

    # ── Méthode / Approche ────────────────────────────────────────────────────
    if qtype == "method":
        method_sents = [s for s in sentences
                        if any(w in s.lower() for w in
                               ["méthode","approche","étape","step","processus","procédure",
                                "phase","d'abord","ensuite","finally","first","then","must"])]
        if method_sents:
            answer = "**Méthode et approche décrites :**\n\n"
            answer += "\n\n".join(f"• {s}" for s in method_sents[:5])
            return {"answer": answer, "sources": method_sents[:5], "mode": "local"}

    # ── Recherche générale par pertinence ─────────────────────────────────────
    scored_p = sorted([(p, _score_text(p, q_words)) for p in paragraphs],
                      key=lambda x: x[1], reverse=True)
    top_p = [(p, sc) for p, sc in scored_p if sc > 0.08][:3]

    if top_p:
        answer = f"**Réponse basée sur le document « {title} » :**\n\n"
        for para, _ in top_p:
            preview = para[:300].rstrip()
            if len(para) > 300:
                preview += "…"
            answer += f"{preview}\n\n"
        return {"answer": answer.strip(), "sources": [p for p, _ in top_p], "mode": "local"}

    # ── Fallback : résumé ─────────────────────────────────────────────────────
    if summary:
        return {
            "answer": (f"Je n'ai pas trouvé de passage directement lié à cette question.\n\n"
                       f"**Voici ce que contient le document « {title} » :**\n\n{summary}"),
            "sources": [], "mode": "local",
        }

    return {
        "answer": "Aucune information pertinente trouvée. Reformulez votre question avec des termes du document.",
        "sources": [], "mode": "local",
    }


# =============================================================================
# Fonction publique
# =============================================================================

def answer_document_question(
    question     : str,
    document_text: str,
    meta         : dict | None = None,
    history      : list[dict] | None = None,
) -> dict:
    meta    = meta    or {}
    history = history or []

    try:
        if not question.strip():
            raise ValueError("Question vide.")
        if not document_text.strip():
            raise ValueError("Texte du document vide.")

        # 1. Groq (priorité)
        if GROQ_API_KEY:
            try:
                answer = _groq_answer(question, document_text, meta, history)
                result = {"answer": answer, "sources": [], "mode": "Groq"}
                log_action("ChatAgent", "answer_question", "success",
                           {"mode": "Groq", "q_len": len(question), "a_len": len(answer)})
                return {"status": "success", **result}
            except Exception as exc:
                log_action("ChatAgent", "groq_error", "warning", {"error": str(exc)})

        # 2. Gemini (fallback)
        if GEMINI_API_KEY:
            try:
                answer = _gemini_answer(question, document_text, meta, history)
                result = {"answer": answer, "sources": [], "mode": "Gemini"}
                log_action("ChatAgent", "answer_question", "success",
                           {"mode": "Gemini", "q_len": len(question), "a_len": len(answer)})
                return {"status": "success", **result}
            except Exception as exc:
                log_action("ChatAgent", "gemini_error", "warning", {"error": str(exc)})

        # 3. Local intelligent
        result = _local_answer(question, document_text, meta)
        log_action("ChatAgent", "answer_question", "success",
                   {"mode": "local", "q_len": len(question), "a_len": len(result["answer"])})
        return {"status": "success", **result}

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log_action("ChatAgent", "answer_question", "error", {"error": err})
        return {"status": "error", "error": err,
                "answer": f"Erreur : {err}", "sources": [], "mode": "error"}
