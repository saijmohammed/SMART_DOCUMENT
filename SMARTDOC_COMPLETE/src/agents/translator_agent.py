"""
src/agents/translator_agent.py — Agent Spécialiste #5 : Traduction.
Rôle    : Traduire le résumé vers la langue cible (FR / EN / AR).
Outil   : translate_text (Tool #5)
"""
from src.tools.translator_tool import translate_text
from src.tools.logger_tool     import log_action

AGENT_NAME = "TranslatorAgent"
AGENT_ROLE = "Multilingual Translation Specialist"
AGENT_GOAL = "Translate document summaries accurately into the requested target language."


def run(text: str, target_language: str = "fr") -> dict:
    """Traduit le texte vers la langue cible."""
    log_action(AGENT_NAME, "run", "info", {
        "role"  : AGENT_ROLE,
        "target": target_language,
    })
    result = translate_text(text, target_language=target_language)
    status = "success" if result.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "mode" : result.get("mode"),
        "chars": len(result.get("translated_text", "")),
        "error": result.get("error"),
    })
    return result
