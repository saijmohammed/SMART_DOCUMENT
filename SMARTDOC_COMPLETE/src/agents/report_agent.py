"""
src/agents/report_agent.py — Agent Spécialiste #6 : Génération de rapport.
Rôle    : Compiler tous les résultats en un rapport PDF professionnel.
Outil   : generate_pdf_report (Tool #6)
Déclenché uniquement après validation humaine (HITL).
"""
from src.tools.report_tool import generate_pdf_report
from src.tools.logger_tool import log_action

AGENT_NAME = "ReportAgent"
AGENT_ROLE = "Report Compilation Specialist"
AGENT_GOAL = "Assemble all analysis outputs into a professional PDF report after HITL validation."


def run(result: dict) -> dict:
    """Génère le rapport PDF général. Ne doit être appelé qu'après la validation HITL."""
    log_action(AGENT_NAME, "run", "info", {"role": AGENT_ROLE})
    report = generate_pdf_report(result)
    status = "success" if report.get("status") == "success" else "error"
    log_action(AGENT_NAME, "run", status, {
        "report_path": report.get("report_path"),
        "error"      : report.get("error"),
    })
    return report
