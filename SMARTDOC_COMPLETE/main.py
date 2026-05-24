"""
main.py — SmartDoc AI CLI.

Lance le pipeline d'analyse en ligne de commande.

Usage
-----
    python main.py <chemin_pdf> [options]

Exemples
--------
    python main.py docs/rapport.pdf
    python main.py docs/paper.pdf --translate --lang en
    python main.py docs/facture.pdf --no-explain
    python main.py docs/cours.pdf --translate --lang ar --json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.agents.orchestrator import SmartDocOrchestrator


# ── Utilitaires d'affichage couleur ──────────────────────────────────────────

def c(text: str, code: str) -> str:
    """Applique un code couleur ANSI."""
    return f"\033[{code}m{text}\033[0m"


def header(title: str) -> None:
    """Affiche un en-tête de section."""
    print("\n" + c("─" * 60, "34"))
    print(c(f"  {title}", "1;36"))
    print(c("─" * 60, "34"))


def kv(key: str, value: str) -> None:
    """Affiche une paire clé-valeur alignée."""
    print(f"  {c(key + ':', '90'):<32} {c(str(value), '97')}")


def wrap(text: str, width: int = 72) -> None:
    """Affiche un texte en respectant la largeur de ligne."""
    words, line = text.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(f"  {line}")
            line = w
        else:
            line += (" " if line else "") + w
    if line:
        print(f"  {line}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmartDoc AI — Analyseur PDF multi-agents (UIR S8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf",
                        help="Chemin vers le fichier PDF à analyser")
    parser.add_argument("--no-explain",
                        dest="explanation",
                        action="store_false",
                        default=True,
                        help="Désactiver l'explication")
    parser.add_argument("--translate",
                        dest="translation",
                        action="store_true",
                        default=False,
                        help="Activer la traduction")
    parser.add_argument("--lang",
                        dest="target_language",
                        default="fr",
                        choices=["fr", "en", "ar"],
                        help="Langue cible de traduction (défaut : fr)")
    parser.add_argument("--json",
                        action="store_true",
                        help="Afficher le résultat brut en JSON")
    args = parser.parse_args()

    # ── Vérification du fichier ───────────────────────────────────────────────
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(c(f"\n  ✗ Fichier introuvable : {pdf_path}", "31"))
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(c(f"\n  ✗ Le fichier doit être un PDF : {pdf_path}", "31"))
        sys.exit(1)

    # ── En-tête ───────────────────────────────────────────────────────────────
    print(c("\n  ⚡  SmartDoc AI — Multi-Agent PDF Intelligence", "1;34"))
    print(c("  UIR S8 · Projet Intégré · v4.0\n", "90"))

    # ── Callback de progression ───────────────────────────────────────────────
    def progress_cb(step: str, status: str, pct: float) -> None:
        ico = {"done": "✓", "running": "⟳", "error": "✗"}.get(status, "·")
        col = {"done": "32", "running": "34", "error": "31"}.get(status, "90")
        bar = "█" * int(pct * 22) + "░" * (22 - int(pct * 22))
        print(f"  {c(ico, col)}  {step:<20}  [{c(bar, '34')}]  {int(pct * 100):3d}%")

    # ── Analyse ───────────────────────────────────────────────────────────────
    orch   = SmartDocOrchestrator(progress_cb=progress_cb)
    result = orch.analyze(
        str(pdf_path),
        do_summary     = True,
        do_explanation = args.explanation,
        do_translation = args.translation,
        target_language= args.target_language,
    )

    # ── Sortie JSON brute ─────────────────────────────────────────────────────
    if args.json:
        # Ne pas inclure raw_text dans la sortie JSON pour la lisibilité
        output = {k: v for k, v in result.items() if k != "raw_text"}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # ── Erreur ────────────────────────────────────────────────────────────────
    if result.get("status") != "success":
        print(c(
            f"\n  ✗ Erreur à l'étape '{result.get('step', '?')}' : "
            f"{result.get('error', 'Erreur inconnue')}",
            "31",
        ))
        sys.exit(1)

    # ── Informations document ─────────────────────────────────────────────────
    header("Informations document")
    doc = result.get("document", {})
    cls = result.get("classification", {})
    kv("Fichier",            doc.get("file_name", "N/A"))
    kv("Pages",              doc.get("page_count", "N/A"))
    kv("Mots",               f"{doc.get('word_count', 0):,}")
    kv("Catégorie (DL)",     cls.get("category", "N/A"))
    kv("Confiance",          f"{cls.get('confidence', 0):.1%}")
    kv("Mode classification",cls.get("mode", "N/A"))

    # ── Résumé ────────────────────────────────────────────────────────────────
    summ = result.get("summary", {})
    if summ.get("summary"):
        header("Résumé détaillé")
        wrap(summ["summary"])

    if summ.get("key_points"):
        header("Points clés")
        for i, p in enumerate(summ["key_points"], 1):
            print(f"  {c(str(i) + '.', '34')} {p[:120]}")

    # ── Explication ───────────────────────────────────────────────────────────
    exp = result.get("explanation", {})
    if exp.get("important_terms"):
        header("Termes importants")
        print("  " + "  ·  ".join(exp["important_terms"][:10]))

    if exp.get("study_questions"):
        header("Questions d'étude")
        for i, q in enumerate(exp["study_questions"], 1):
            print(f"  {c(str(i) + '.', '34')} {q}")

    # ── Traduction ────────────────────────────────────────────────────────────
    trn = result.get("translation", {})
    if trn.get("translated_text"):
        lang_map = {"fr": "Français", "en": "English", "ar": "Arabe"}
        lang     = lang_map.get(trn.get("target_language", ""), "Inconnu")
        header(f"Traduction ({lang})")
        wrap(trn["translated_text"][:800])

    # ── HITL en CLI ───────────────────────────────────────────────────────────
    header("Validation humaine (HITL)")
    print("  Vérifiez les résultats ci-dessus.")
    print(c("  La génération du rapport PDF requiert votre validation.", "33"))

    try:
        ans = input(c("\n  Valider et générer le rapport PDF ? [o/N] : ", "33"))
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    if ans.strip().lower() in ("o", "y", "oui", "yes"):
        note = ""
        try:
            note = input(c("  Note de validation (optionnel) : ", "90"))
        except (EOFError, KeyboardInterrupt):
            pass

        print(c("  Génération du rapport en cours…", "90"))
        report = orch.generate_report_after_human_validation(result, validator_note=note)

        if report.get("status") == "success":
            print(c(f"\n  ✓ Rapport PDF sauvegardé :\n    {report['report_path']}", "32"))
        else:
            print(c(f"\n  ✗ Erreur rapport : {report.get('error')}", "31"))
    else:
        print(c("  Génération du rapport ignorée.", "90"))

    print(c("\n  Terminé.\n", "90"))


if __name__ == "__main__":
    main()
