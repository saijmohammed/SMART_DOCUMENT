"""
src/agents/explainer_agent.py — Agent Spécialiste #4 : Explication.
Rôle    : Explication simple, termes clés, questions d'étude.
Outil   : explain_text (Tool #4)
"""
from src.tools.nlp_tool    import explain_text
from src.tools.logger_tool import log_action

AGENT_NAME = "ExplainerAgent"
AGENT_ROLE = "Content Explanation Specialist"
AGENT_GOAL = "Make complex document content accessible with plain-language explanations."


def run(text: str, category: str = "") -> dict:
    """Génère une explication simple, des termes et des questions d'étude."""
    log_action(AGENT_NAME, "run", "info", {
        "role"    : AGENT_ROLE,
        "category": category,
    })
    result = explain_text(text, category=category)
    status = "success" if result.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "mode" : result.get("mode"),
        "level": result.get("reading_level"),
        "error": result.get("error"),
    })
    return result
