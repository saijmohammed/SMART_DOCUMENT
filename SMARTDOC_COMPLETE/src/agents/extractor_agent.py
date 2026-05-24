"""
src/agents/extractor_agent.py — Agent Spécialiste #1 : Extraction PDF.
Rôle    : Extraire le texte brut et les métadonnées des documents PDF.
Outil   : extract_pdf_text (Tool #1)
"""
from src.tools.extractor_tool import extract_pdf_text
from src.tools.logger_tool    import log_action

AGENT_NAME = "ExtractorAgent"
AGENT_ROLE = "PDF Extraction Specialist"
AGENT_GOAL = "Extract all text and metadata from PDF documents accurately and robustly."


def run(pdf_path: str) -> dict:
    """Lance l'extraction du PDF. Retourne un dict avec status/text/pages."""
    log_action(AGENT_NAME, "run", "info", {
        "pdf_path": pdf_path,
        "role"    : AGENT_ROLE,
    })
    result = extract_pdf_text(pdf_path)
    status = "success" if result.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "pages": result.get("page_count"),
        "words": result.get("word_count"),
        "error": result.get("error"),
    })
    return result
