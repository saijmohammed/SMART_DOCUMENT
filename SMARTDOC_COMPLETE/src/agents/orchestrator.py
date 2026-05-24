"""
src/agents/orchestrator.py — Orchestrateur SmartDoc AI.

Coordonne tous les agents spécialistes dans un pipeline séquentiel
et impose le checkpoint Human-in-the-Loop (HITL) avant la génération du rapport.

Pipeline
--------
ExtractorAgent → ClassifierAgent → SummarizerAgent
    → ExplainerAgent (optionnel) → TranslatorAgent (optionnel)
    ──── CHECKPOINT VALIDATION HUMAINE ────
    → ReportAgent

Conformité cahier des charges UIR S8 :
    ✔ 2+ agents spécialistes + 1 orchestrateur
    ✔ Modèle DL comme outil fonctionnel (ClassifierAgent — PyTorch)
    ✔ Gestion d'erreurs robuste — le pipeline ne crashe jamais
    ✔ Logging JSON horodaté (logger_tool)
    ✔ HITL obligatoire avant la génération du rapport
"""
from typing import Callable

from src.agents import (
    classifier_agent,
    explainer_agent,
    extractor_agent,
    report_agent,
    summarizer_agent,
    translator_agent,
)
from src.tools.logger_tool import log_action

ProgressCB = Callable[[str, str, float], None] | None


class SmartDocOrchestrator:
    """
    Orchestre le pipeline multi-agents SmartDoc.

    Paramètres
    ----------
    progress_cb : callable optionnel (step_name, status, pct_completion)
                  — utilisé par l'interface Streamlit pour les mises à jour temps réel.
    """

    PIPELINE_STEPS = [
        "extraction",
        "classification",
        "summarisation",
        "explanation",
        "translation",
        "human_validation",
        "report",
    ]

    def __init__(self, progress_cb: ProgressCB = None):
        self._cb = progress_cb

    def _progress(self, step: str, status: str, pct: float) -> None:
        """Appelle le callback de progression si défini. Ne lève jamais d'exception."""
        if self._cb:
            try:
                self._cb(step, status, min(max(float(pct), 0.0), 1.0))
            except Exception:
                pass

    # ── Pipeline principal d'analyse ──────────────────────────────────────────

    def analyze(
        self,
        pdf_path       : str,
        *,
        do_summary     : bool = True,
        do_explanation : bool = True,
        do_translation : bool = False,
        target_language: str  = "fr",
    ) -> dict:
        """
        Exécute le pipeline d'analyse complet sans générer le rapport final.
        Le rapport est différé jusqu'après validation humaine explicite.

        Paramètres
        ----------
        pdf_path        : chemin absolu ou relatif vers le fichier PDF
        do_summary      : activer le résumé (toujours True dans l'interface)
        do_explanation  : activer l'explication
        do_translation  : activer la traduction
        target_language : code langue cible ('fr', 'en', 'ar')

        Retourne
        --------
        dict — résultat complet avec status == "success" ou dict d'erreur
        """
        log_action("Orchestrator", "analyze_start", "info", {
            "pdf_path"       : str(pdf_path),
            "do_summary"     : do_summary,
            "do_explanation" : do_explanation,
            "do_translation" : do_translation,
            "target_language": target_language,
        })

        # ── Étape 1 : Extraction ──────────────────────────────────────────────
        self._progress("extraction", "running", 0.05)
        extraction = extractor_agent.run(str(pdf_path))
        if extraction.get("status") != "success":
            return self._error("extraction", extraction.get("error", "Extraction échouée"))
        self._progress("extraction", "done", 0.20)
        text = extraction["text"]

        # ── Étape 2 : Classification (modèle Deep Learning) ───────────────────
        self._progress("classification", "running", 0.22)
        classification = classifier_agent.run(text)
        if classification.get("status") != "success":
            return self._error("classification",
                               classification.get("error", "Classification échouée"))
        self._progress("classification", "done", 0.40)

        # ── Étape 3 : Résumé ──────────────────────────────────────────────────
        if do_summary:
            self._progress("summarisation", "running", 0.42)
            summary = summarizer_agent.run(text)
            if summary.get("status") != "success":
                return self._error("summarisation",
                                   summary.get("error", "Résumé échoué"))
            self._progress("summarisation", "done", 0.60)
        else:
            summary = {"status": "skipped", "summary": "", "key_points": [], "mode": "skipped"}

        # ── Étape 4 : Explication ─────────────────────────────────────────────
        if do_explanation:
            self._progress("explanation", "running", 0.62)
            explanation = explainer_agent.run(text, classification.get("category", ""))
            if explanation.get("status") != "success":
                return self._error("explanation",
                                   explanation.get("error", "Explication échouée"))
            self._progress("explanation", "done", 0.76)
        else:
            explanation = {"status": "skipped"}

        # ── Étape 5 : Traduction (optionnelle) ────────────────────────────────
        translation = {"status": "skipped"}
        if do_translation:
            self._progress("translation", "running", 0.78)
            source      = summary.get("summary") or text[:2_000]
            translation = translator_agent.run(source, target_language)
            if translation.get("status") != "success":
                return self._error("translation",
                                   translation.get("error", "Traduction échouée"))
            self._progress("translation", "done", 0.90)

        # ── Construction du résultat complet ──────────────────────────────────
        self._progress("human_validation", "pending", 0.92)

        result = {
            "status"          : "success",
            "document"        : {
                "file_name"   : extraction["file_name"],
                "page_count"  : extraction["page_count"],
                "word_count"  : extraction.get("word_count", 0),
                "char_count"  : extraction.get("char_count", 0),
                "text_preview": text[:1_500],
            },
            "classification"  : classification,
            "summary"         : summary,
            "explanation"     : explanation,
            "translation"     : translation,
            "human_validation": "⏳ En attente de validation humaine.",
            "raw_text"        : text,
        }

        log_action("Orchestrator", "analyze_complete", "success", {
            "file"    : extraction["file_name"],
            "category": classification.get("category"),
            "mode_dl" : classification.get("mode"),
        })
        return result

    # ── Checkpoint HITL + génération rapport ──────────────────────────────────

    def generate_report_after_human_validation(
        self,
        result        : dict,
        validator_note: str = "",
    ) -> dict:
        """
        Checkpoint Human-in-the-Loop.
        Doit être appelé explicitement par l'opérateur (bouton UI ou CLI)
        avant la génération du rapport final.

        Paramètres
        ----------
        result         : dict sortie de analyze()
        validator_note : str commentaire humain optionnel
        """
        if result.get("status") != "success":
            return {
                "status": "error",
                "error" : "Résultat d'analyse invalide — rapport impossible.",
            }

        from datetime import datetime, timezone
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        result["human_validation"] = (
            f"✅ Validé par l'opérateur humain le {ts}"
            + (f" — Note : {validator_note}" if validator_note else "")
        )

        log_action("Orchestrator", "hitl_checkpoint", "success", {
            "timestamp": ts,
            "note"     : validator_note,
        })

        self._progress("report", "running", 0.95)
        report = report_agent.run(result)
        if report.get("status") == "success":
            self._progress("report", "done", 1.0)
        else:
            self._progress("report", "error", 0.96)
        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _error(step: str, message: str) -> dict:
        log_action("Orchestrator", f"error_{step}", "error", {"error": message})
        return {"status": "error", "step": step, "error": message}
