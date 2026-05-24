"""
src/tools/extractor_tool.py — Outil d'extraction PDF (Tool #1).

Schéma d'entrée  : { "pdf_path": str }
Schéma de sortie : {
    "status"     : "success" | "error",
    "file_name"  : str,
    "page_count" : int,
    "text"       : str,
    "char_count" : int,
    "word_count" : int,
    "pages"      : list[dict],
    "error"      : str  (uniquement en cas d'erreur)
}
"""
from pathlib import Path

from src.tools.logger_tool import log_action
from src.utils.pdf_reader  import read_pdf


def extract_pdf_text(pdf_path: str | Path) -> dict:
    """
    Extrait le texte complet d'un PDF.
    Ne lève jamais d'exception — toutes les erreurs sont renvoyées dans le dict résultat.
    """
    try:
        if not pdf_path:
            raise ValueError("pdf_path ne doit pas être vide.")

        result = read_pdf(pdf_path)

        if not result["text"].strip():
            raise ValueError(
                "Aucun texte lisible trouvé dans ce PDF. "
                "Le fichier est peut-être un PDF scanné (image). "
                "L'OCR n'est pas supporté dans cette version."
            )

        log_action("ExtractorAgent", "extract_pdf_text", "success", {
            "file"      : result["file_name"],
            "pages"     : result["page_count"],
            "word_count": result["word_count"],
            "char_count": result["char_count"],
        })
        return {"status": "success", **result}

    except FileNotFoundError as exc:
        msg = str(exc)
        log_action("ExtractorAgent", "extract_pdf_text", "error", {"error": msg})
        return {"status": "error", "error": msg}

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        log_action("ExtractorAgent", "extract_pdf_text", "error", {"error": msg})
        return {"status": "error", "error": msg}
