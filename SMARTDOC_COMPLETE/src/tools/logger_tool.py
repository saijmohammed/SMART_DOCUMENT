"""
src/tools/logger_tool.py
Logger JSON thread-safe pour SmartDoc AI.
Chaque action d'agent est enregistrée avec horodatage ISO dans logs/actions.json.
Exigence cahier des charges UIR S8 : logging JSON horodaté.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import LOG_PATH

_lock = threading.Lock()


def log_action(
    agent  : str,
    action : str,
    status : str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Ajoute un enregistrement structuré dans logs/actions.json.

    Paramètres
    ----------
    agent   : nom de l'agent ou composant (ex : 'ExtractorAgent')
    action  : action réalisée             (ex : 'extract_pdf_text')
    status  : 'success' | 'error' | 'warning' | 'info'
    details : contexte arbitraire — doit être sérialisable en JSON
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "agent"    : agent,
        "action"   : action,
        "status"   : status,
        "details"  : details or {},
    }
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        data: list = []
        if path.exists() and path.stat().st_size > 0:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        data.append(record)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_logs() -> list[dict]:
    """Retourne tous les enregistrements de log (ordre chronologique)."""
    path = Path(LOG_PATH)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def clear_logs() -> None:
    """Vide le fichier de log (utilisé lors du reset de l'interface)."""
    path = Path(LOG_PATH)
    if path.exists():
        path.write_text("[]", encoding="utf-8")
