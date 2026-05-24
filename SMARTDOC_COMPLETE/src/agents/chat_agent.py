"""
src/agents/chat_agent.py — Agent Spécialiste #7 : Chat documentaire.
Rôle    : Répondre aux questions en langage naturel sur le document analysé.
Outil   : answer_document_question (Tool #7)
"""
from src.tools.chat_tool   import answer_document_question
from src.tools.logger_tool import log_action

AGENT_NAME = "ChatAgent"
AGENT_ROLE = "Document Q&A Specialist"
AGENT_GOAL = "Answer user questions accurately based solely on the analysed document content."


def run(
    question     : str,
    document_text: str,
    meta         : dict | None = None,
    history      : list[dict] | None = None,
) -> dict:
    """Répond à une question sur le document. Retourne {status, answer, sources, mode}."""
    log_action(AGENT_NAME, "run", "info", {
        "role"    : AGENT_ROLE,
        "q_len"   : len(question),
        "has_gemini": bool(__import__('src.utils.config', fromlist=['GEMINI_API_KEY']).GEMINI_API_KEY),
    })
    result = answer_document_question(question, document_text, meta=meta, history=history)
    status = "success" if result.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "mode" : result.get("mode"),
        "error": result.get("error"),
    })
    return result
