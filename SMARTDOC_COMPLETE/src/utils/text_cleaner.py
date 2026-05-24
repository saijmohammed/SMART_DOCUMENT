"""
src/utils/text_cleaner.py
Utilitaires de nettoyage, découpage et troncature de texte.
"""
import re
import unicodedata


def clean_text(text: str) -> str:
    """Nettoie le texte : supprime les caractères parasites, normalise les espaces."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 3_500) -> list[str]:
    """Découpe le texte en blocs de max_chars caractères sur les paragraphes."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current.strip())
            current = para[:max_chars]
    if current:
        chunks.append(current.strip())
    return chunks


def truncate(text: str, max_chars: int = 5_000) -> str:
    """Troncature stricte pour les appels API."""
    return text[:max_chars] if len(text) > max_chars else text
