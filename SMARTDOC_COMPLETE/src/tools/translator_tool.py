"""
src/tools/translator_tool.py — Outil de traduction (Tool #5).

Supporte 12 langues. Priorité :
  1. Gemini API  — si GEMINI_API_KEY est défini
  2. deep_translator (GoogleTranslator)
  3. Message de fallback informatif

Schéma d'entrée  : { "text": str, "target_language": str }
Schéma de sortie : {
    "status"         : "success" | "error",
    "translated_text": str,
    "source_text"    : str,
    "target_language": str,
    "mode"           : "gemini" | "deep_translator" | "fallback",
}
"""
from src.tools.logger_tool  import log_action
from src.utils.config       import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.text_cleaner import truncate

# ── Toutes les langues supportées ────────────────────────────────────────────

ALL_LANGUAGES: dict[str, dict] = {
    "fr": {"name": "Français",    "flag": "🇫🇷", "dir": "ltr"},
    "en": {"name": "English",     "flag": "🇬🇧", "dir": "ltr"},
    "ar": {"name": "العربية",     "flag": "🇸🇦", "dir": "rtl"},
    "es": {"name": "Español",     "flag": "🇪🇸", "dir": "ltr"},
    "de": {"name": "Deutsch",     "flag": "🇩🇪", "dir": "ltr"},
    "it": {"name": "Italiano",    "flag": "🇮🇹", "dir": "ltr"},
    "pt": {"name": "Português",   "flag": "🇵🇹", "dir": "ltr"},
    "zh": {"name": "中文",         "flag": "🇨🇳", "dir": "ltr"},
    "ja": {"name": "日本語",       "flag": "🇯🇵", "dir": "ltr"},
    "ru": {"name": "Русский",     "flag": "🇷🇺", "dir": "ltr"},
    "tr": {"name": "Türkçe",      "flag": "🇹🇷", "dir": "ltr"},
    "nl": {"name": "Nederlands",  "flag": "🇳🇱", "dir": "ltr"},
}

# Noms Gemini complets
_GEMINI_LANG_NAMES = {
    "fr": "French (Français)",
    "en": "English",
    "ar": "Arabic (العربية)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "zh": "Chinese Simplified (中文)",
    "ja": "Japanese (日本語)",
    "ru": "Russian (Русский)",
    "tr": "Turkish (Türkçe)",
    "nl": "Dutch (Nederlands)",
}


def get_lang_info(code: str) -> dict:
    """Retourne les infos d'une langue par son code."""
    return ALL_LANGUAGES.get(code, {"name": code, "flag": "🌐", "dir": "ltr"})


def get_lang_label(code: str) -> str:
    """Retourne le label affiché d'une langue."""
    info = ALL_LANGUAGES.get(code, {})
    return f"{info.get('flag','')} {info.get('name', code)}"


def _gemini_translate(text: str, target_language: str) -> str:
    from google import genai  # type: ignore
    client    = genai.Client(api_key=GEMINI_API_KEY)
    lang_name = _GEMINI_LANG_NAMES.get(target_language, target_language)
    prompt    = (
        f"Translate the following text into {lang_name}. "
        "Preserve the structure and formatting. "
        "Return ONLY the translated text, without any preamble or explanation.\n\n"
        f"{truncate(text, 5_000)}"
    )
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()


def _deep_translate(text: str, target_language: str) -> str:
    from deep_translator import GoogleTranslator  # type: ignore
    # deep_translator utilise des codes langue légèrement différents
    code_map = {"zh": "zh-CN"}
    code = code_map.get(target_language, target_language)
    return GoogleTranslator(source="auto", target=code).translate(
        truncate(text, 4_500)
    )


def translate_text(text: str, target_language: str = "fr") -> dict:
    """
    Traduit le texte vers la langue cible.
    Essaie Gemini, puis deep_translator, puis message informatif.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Texte d'entrée vide.")

        translated = ""
        mode       = "fallback"

        if GEMINI_API_KEY:
            try:
                translated = _gemini_translate(text, target_language)
                mode = "gemini"
            except Exception as exc:
                log_action("TranslatorAgent", "gemini_error", "warning",
                           {"error": str(exc)})

        if not translated:
            try:
                translated = _deep_translate(text, target_language)
                mode = "deep_translator"
            except Exception as exc:
                log_action("TranslatorAgent", "deep_translator_error", "warning",
                           {"error": str(exc)})

        if not translated:
            lang_label = get_lang_label(target_language)
            translated = (
                f"[Traduction vers {lang_label} indisponible]\n\n"
                "Pour activer la traduction, ajoutez GEMINI_API_KEY dans votre .env "
                "ou installez : pip install deep-translator\n\n"
                f"Texte source :\n{text[:600]}"
            )
            mode = "fallback"

        log_action("TranslatorAgent", "translate_text", "success", {
            "target_language": target_language,
            "mode"           : mode,
            "chars"          : len(translated),
        })
        return {
            "status"         : "success",
            "translated_text": translated,
            "source_text"    : text[:2_000],
            "target_language": target_language,
            "mode"           : mode,
        }

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log_action("TranslatorAgent", "translate_text", "error", {"error": err})
        return {"status": "error", "error": err}
