"""
src/tools/classifier_tool.py — Outil de classification (Tool #2 — modèle PyTorch).

Architecture : TF-IDF → réseau FC avec BatchNorm + Dropout.
Fallback      : règles heuristiques si les fichiers modèle sont absents.

Schéma d'entrée  : { "text": str }
Schéma de sortie : {
    "status"    : "success" | "error",
    "category"  : str,
    "confidence": float,
    "mode"      : "pytorch_model" | "heuristic_fallback",
    "all_scores": dict[str, float]
}
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.tools.logger_tool import log_action
from src.utils.config import MODELS_DIR


# ── Définition du modèle (identique à train_model.py) ────────────────────────

class SimpleDocumentClassifier(nn.Module):
    """
    Classificateur de texte entièrement connecté.
    Entrée  : vecteur TF-IDF de dimension input_dim.
    Sortie  : logits de dimension num_classes.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Règles heuristiques (fallback si le modèle est absent) ───────────────────

_RULES: dict[str, list[str]] = {
    "Course / Academic"       : ["chapter", "cours", "theorem", "proof", "definition",
                                 "exercise", "exercice", "lecture", "syllabus", "assignment",
                                 "étudiant", "cours", "séance", "devoir"],
    "Research Paper"          : ["doi", "references", "journal", "experiment", "dataset",
                                 "hypothesis", "methodology", "abstract", "findings",
                                 "results", "conclusion", "study"],
    "Report"                  : ["report", "rapport", "objective", "recommendation",
                                 "executive summary", "methodology", "conclusion",
                                 "résultats", "analyse", "bilan"],
    "Business / Invoice"      : ["invoice", "facture", "total", "amount", "payment",
                                 "tax", "customer", "purchase", "order", "price",
                                 "montant", "tva", "client"],
    "Administrative"          : ["attestation", "demande", "certificat", "signature",
                                 "administration", "official", "bureau", "ministry",
                                 "formulaire", "référence", "arrêté"],
    "Legal / Contract"        : ["contract", "contrat", "clause", "agreement", "party",
                                 "liability", "jurisdiction", "warrant", "legal",
                                 "loi", "droit", "partie"],
    "Medical / Health"        : ["patient", "diagnosis", "treatment", "symptom",
                                 "clinical", "physician", "hospital", "medical",
                                 "santé", "maladie", "médicament"],
    "Technical / Engineering" : ["circuit", "algorithm", "specification", "component",
                                 "engineering", "design", "system", "architecture",
                                 "code", "logiciel", "réseau", "protocole"],
}


def _heuristic_classify(text: str) -> dict:
    t      = text.lower()
    scores = {lbl: sum(1 for kw in kws if kw in t) for lbl, kws in _RULES.items()}
    best   = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_n = scores[best]
    conf   = min(0.88, 0.42 + 0.08 * best_n) if best_n > 0 else 0.42
    total  = sum(scores.values()) or 1
    return {
        "category"  : best,
        "confidence": round(conf, 4),
        "mode"      : "heuristic_fallback",
        "all_scores": {k: round(v / total, 3) for k, v in scores.items()},
    }


# ── Chargement des artefacts ──────────────────────────────────────────────────

def _load_artifacts() -> tuple[Any, dict]:
    import joblib
    import pickle  # noqa: F401

    def _load(path: Path):
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as fh:
                return pickle.load(fh)

    vectorizer  = _load(MODELS_DIR / "vectorizer.pkl")
    id_to_label = _load(MODELS_DIR / "id_to_label.pkl")
    return vectorizer, id_to_label


# ── Fonction publique ─────────────────────────────────────────────────────────

def classify_document(text: str) -> dict:
    """
    Classifie le texte avec le modèle PyTorch si disponible,
    sinon utilise les règles heuristiques.

    Ne lève jamais d'exception — toutes les erreurs sont renvoyées dans le dict résultat.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Le texte d'entrée est vide — impossible de classifier.")

        model_path      = MODELS_DIR / "document_classifier.pth"
        vectorizer_path = MODELS_DIR / "vectorizer.pkl"
        labels_path     = MODELS_DIR / "id_to_label.pkl"

        # ── Fallback heuristique si les fichiers modèle sont absents ─────────
        if not (model_path.exists() and vectorizer_path.exists() and labels_path.exists()):
            result = _heuristic_classify(text)
            log_action("ClassifierAgent", "classify_document", "success", result)
            return {"status": "success", **result}

        # ── Chemin PyTorch ────────────────────────────────────────────────────
        vectorizer, id_to_label = _load_artifacts()
        X = vectorizer.transform([text]).toarray().astype(np.float32)

        model = SimpleDocumentClassifier(X.shape[1], len(id_to_label))
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True)
        )
        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(X))
            probs  = torch.softmax(logits, dim=1).numpy()[0]

        pred_id    = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        category   = str(id_to_label[pred_id])
        all_scores = {
            str(id_to_label[i]): round(float(probs[i]), 4)
            for i in range(len(id_to_label))
        }

        result = {
            "category"  : category,
            "confidence": round(confidence, 4),
            "mode"      : "pytorch_model",
            "all_scores": all_scores,
        }
        log_action("ClassifierAgent", "classify_document", "success", result)
        return {"status": "success", **result}

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        log_action("ClassifierAgent", "classify_document", "error", {"error": err})
        # Fallback ultime — ne jamais faire crasher le pipeline
        try:
            result = _heuristic_classify(text)
            return {"status": "success", **result, "fallback_reason": err}
        except Exception:
            return {"status": "error", "error": err}
