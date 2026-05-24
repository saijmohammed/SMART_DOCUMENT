"""
src/agents/summarizer_agent.py — Agent Spécialiste #3 : Résumé.
Rôle    : Générer un résumé détaillé et des points clés.
Outil   : summarize_text (Tool #3)
"""
from src.tools.summarizer_tool import summarize_text
from src.tools.logger_tool     import log_action

AGENT_NAME = "SummarizerAgent"
AGENT_ROLE = "Document Summarisation Specialist"
AGENT_GOAL = "Generate a rich summary and structured key points from the document."


def run(text: str, max_sentences: int = 7) -> dict:
    """Résume le texte du document. Retourne le résumé + points clés."""
    log_action(AGENT_NAME, "run", "info", {
        "role"    : AGENT_ROLE,
        "text_len": len(text),
    })
    result = summarize_text(text, max_sentences=max_sentences)
    status = "success" if result.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "mode" : result.get("mode"),
        "error": result.get("error"),
    })
    return result
