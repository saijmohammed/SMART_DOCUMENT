"""
src/utils/pdf_reader.py
Extraction robuste de texte depuis un PDF via PyMuPDF (fitz).
"""
from pathlib import Path

import fitz  # PyMuPDF

from src.utils.text_cleaner import clean_text


def read_pdf(pdf_path: str | Path) -> dict:
    """
    Extrait tout le texte d'un PDF page par page.

    Retourne
    --------
    dict :
        file_name  : str
        page_count : int
        text       : str   — texte complet nettoyé
        pages      : list[dict]  — texte par page
        char_count : int
        word_count : int
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Fichier attendu en .pdf, reçu : {pdf_path.suffix}")

    pages: list[dict] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            raw     = page.get_text("text") or ""
            cleaned = clean_text(raw)
            pages.append({"page": i, "text": cleaned, "char_count": len(cleaned)})

    full_text  = clean_text("\n\n".join(p["text"] for p in pages))
    word_count = len(full_text.split())

    return {
        "file_name" : pdf_path.name,
        "page_count": len(pages),
        "text"      : full_text,
        "pages"     : pages,
        "char_count": len(full_text),
        "word_count": word_count,
    }
