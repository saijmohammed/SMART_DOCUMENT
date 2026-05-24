"""
src/agents/classifier_agent.py — Agent Spécialiste #2 : Classification.
Rôle    : Classifier le document en 8 catégories via modèle PyTorch.
Outil   : classify_document (Tool #2 — modèle Deep Learning)
"""
from src.tools.classifier_tool import classify_document
from src.tools.logger_tool     import log_action

AGENT_NAME = "ClassifierAgent"
AGENT_ROLE = "Deep Learning Document Classifier"
AGENT_GOAL = "Classify documents using the trained PyTorch model with confidence scoring."


def run(text: str) -> dict:
    """Classifie le texte du document. Retourne catégorie + confiance."""
    log_action(AGENT_NAME, "run", "info", {
        "role"    : AGENT_ROLE,
        "text_len": len(text),
    })
    result = classify_document(text)
    status = "success" if result.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "category"  : result.get("category"),
        "confidence": result.get("confidence"),
        "mode"      : result.get("mode"),
        "error"     : result.get("error"),
    })
    return result
