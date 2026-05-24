"""
train_model.py — Entraînement du classificateur de documents (PyTorch).

Pipeline :
  1. Génération d'un dataset synthétique équilibré (8 catégories)
  2. TF-IDF vectorisation
  3. Entraînement réseau FC (BatchNorm + Dropout)
  4. Évaluation : accuracy, rapport de classification, matrice de confusion
  5. Sauvegarde : models/document_classifier.pth, vectorizer.pkl, id_to_label.pkl
                  models/evaluation_metrics.json, models/confusion_matrix.png

Usage :
    python train_model.py
"""

import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 4_000
EPOCHS             = 40
LR                 = 1e-3
BATCH_SIZE         = 32
RANDOM_STATE       = 42
SAMPLES_PER_CLASS  = 120   # samples synthétiques par catégorie

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ── Dataset synthétique ───────────────────────────────────────────────────────

TEMPLATES: dict[str, list[str]] = {
    "Course / Academic": [
        "This lecture covers the fundamental theorems of {topic}. Students are expected to solve the exercises and submit assignments by the deadline.",
        "Chapter {n}: Introduction to {topic}. Learning objectives: understand the key concepts, apply the methods, and analyze case studies.",
        "Cours magistral — Séance {n} : {topic}. Objectifs pédagogiques : maîtriser les définitions, résoudre les exercices, réaliser le devoir.",
        "Syllabus for the course on {topic}. Weekly schedule, grading policy, and required readings are described below.",
        "TP {n} — {topic} : implémentation et évaluation. Rendu attendu avant la prochaine séance.",
        "Exam preparation guide for {topic}. Review theorems, proofs, and problem-solving strategies covered in lectures.",
        "Assignment {n}: Apply the concepts of {topic} to the following dataset. Submit a Jupyter notebook with your analysis.",
        "Definition: A {topic} is defined as... Theorem: Under the following conditions... Proof by induction...",
        "Travaux dirigés — {topic}. Exercice 1 : démontrer que... Exercice 2 : calculer... Exercice 3 : analyser...",
        "Course notes on {topic}. Key concepts reviewed: algorithm complexity, data structures, and implementation strategies.",
    ],
    "Research Paper": [
        "Abstract: In this paper, we propose a novel approach to {topic} using deep learning. Our experiments on benchmark datasets demonstrate state-of-the-art performance.",
        "We present a methodology for {topic} based on transformer architectures. The dataset consists of {n}k samples collected from diverse sources.",
        "Introduction: {topic} has attracted significant research interest in recent years. Prior work [1,2,3] has shown promising results, but limitations remain.",
        "Experimental results show that our approach achieves {n}% accuracy on the test set, outperforming baseline methods by a significant margin.",
        "Related work: Several studies have investigated {topic}. Smith et al. [2023] proposed... whereas Jones et al. [2024] demonstrated...",
        "Conclusion: We have proposed a novel framework for {topic}. Future work will focus on scalability and robustness to domain shift.",
        "DOI: 10.1234/journal.{n}. Published in the Journal of {topic}. Received: January 2025. Accepted: March 2025.",
        "Hypothesis: We hypothesize that {topic} can be modeled using a hierarchical attention mechanism. We validate this claim through rigorous experimentation.",
        "Dataset: We collected {n},000 annotated samples for {topic}. The dataset is split 80/10/10 for train, validation, and test respectively.",
        "Methodology: The proposed framework for {topic} consists of three stages: feature extraction, representation learning, and classification.",
    ],
    "Report": [
        "Executive Summary: This report presents the findings of our analysis on {topic}. Key recommendations include improving efficiency and reducing costs.",
        "Rapport d'analyse — {topic}. Objectif : évaluer la performance et proposer des recommandations stratégiques pour l'amélioration du système.",
        "Section 1: Introduction and context. Section 2: Methodology. Section 3: Results. Section 4: Recommendations for {topic}.",
        "Findings: The analysis of {topic} reveals three major issues. First... Second... Third... Recommendations are provided in Section 4.",
        "Quarterly report on {topic}. KPIs monitored: revenue growth, customer satisfaction, operational efficiency. Period: Q{n} 2025.",
        "Bilan de projet — {topic}. État d'avancement : 75% complété. Points bloquants identifiés. Plan d'action recommandé.",
        "Audit report on {topic}. Scope: internal processes and compliance. Findings: 3 major risks, 7 minor observations. Action plan attached.",
        "Technical report: performance evaluation of {topic}. Benchmark results, comparative analysis, and improvement roadmap.",
        "Annual report {n}: overview of activities, financial highlights, and strategic outlook for {topic} division.",
        "Rapport de stage — {topic}. Présentation de l'entreprise, missions réalisées, compétences acquises, bilan personnel.",
    ],
    "Business / Invoice": [
        "INVOICE #{n} — Date: 15/05/2025. Bill To: Client Corp. Item: {topic} services. Subtotal: $2,500. Tax (20%): $500. Total: $3,000.",
        "Facture N°{n} — Objet : prestation {topic}. Montant HT : 1 200 €. TVA 20% : 240 €. Montant TTC : 1 440 €. Échéance : 30 jours.",
        "Purchase Order #{n}: {topic} equipment. Quantity: {n} units. Unit price: $150. Total amount: ${n}00. Payment terms: Net 30.",
        "Quote for {topic} project. Total estimated cost: $15,000. Breakdown: labor 60%, materials 30%, overhead 10%. Valid for 30 days.",
        "Receipt: Payment received for {topic} subscription. Amount: €99/month. Invoice #{n}. Thank you for your business.",
        "Contract invoice for {topic} consulting services. Hours: {n}h at $120/h. Total: ${n}20. VAT exempt. Wire transfer details enclosed.",
        "Bon de commande — {topic}. Référence produit: REF-{n}. Quantité: {n}. Prix unitaire HT: 45€. Remise 10%: -4.50€. Total TTC: 48.60€.",
        "Devis N°{n} — {topic}. Main d'œuvre: 500€. Matériaux: 300€. Déplacement: 50€. Total HT: 850€. TVA: 170€. Total TTC: 1020€.",
        "Monthly billing statement for {topic} services. Account: #{n}. Previous balance: $0. New charges: $299. Due date: 30/06/2025.",
        "Payment confirmation: Transaction #{n} for {topic}. Amount: $450 charged to card ending 4242. Status: Approved.",
    ],
    "Administrative": [
        "ATTESTATION DE SCOLARITÉ — L'Université certifie que M./Mme {topic} est régulièrement inscrit(e) en {n}ème année pour l'année 2025-2026.",
        "Certificat de travail : La société {topic} atteste que M./Mme ... a occupé le poste de ... du ... au ... avec satisfaction.",
        "Formulaire de demande de congé. Nom : ... Service : {topic}. Dates demandées : du ... au ... Motif : ... Signature du responsable.",
        "Note de service N°{n} — Objet : {topic}. La direction informe l'ensemble du personnel que... À compter du... Cette mesure...",
        "Administrative form: Application for {topic} authorization. Required documents: ID copy, proof of address, signed declaration.",
        "Arrêté préfectoral N°{n} relatif à {topic}. Vu le décret... Considérant que... Le préfet arrête : Article 1er...",
        "Convocation à la réunion du {n}/06/2025. Ordre du jour : {topic}. Lieu : Salle de conférence. Heure : 10h00. Présence obligatoire.",
        "Fiche d'inscription — Formation {topic}. Nom : ... Prénom : ... Établissement : ... Niveau : ... Pièces à fournir : ...",
        "Official notice: renewal of {topic} permit. Reference: ADM-{n}. Please submit required documents by the deadline indicated.",
        "Circular memo regarding {topic} policy update. Effective date: 01/07/2025. All departments are requested to comply with the new guidelines.",
    ],
    "Legal / Contract": [
        "SERVICE AGREEMENT between {topic} Inc. (\"Provider\") and Client Corp. (\"Client\"). This Agreement governs the provision of services.",
        "CONTRAT DE TRAVAIL — Entre la société {topic} et M./Mme ... CDI à temps plein. Rémunération brute mensuelle : ... Article 1 : Fonctions...",
        "WHEREAS, the parties wish to enter into an agreement regarding {topic}; NOW THEREFORE, in consideration of the mutual covenants herein...",
        "Clause de confidentialité : Le prestataire s'engage à ne pas divulguer les informations relatives à {topic} à des tiers sans autorisation.",
        "Liability clause: In no event shall {topic} be liable for indirect, incidental, or consequential damages arising from this agreement.",
        "Non-disclosure agreement (NDA) for {topic} project. Duration: 3 years. Jurisdiction: Paris, France. Governing law: French law.",
        "Bail commercial N°{n} — Objet : location des locaux situés... Durée : 3/6/9 ans. Loyer mensuel HT : ... Dépôt de garantie : ...",
        "Terms and Conditions for {topic} platform. By using this service, you agree to the following terms. Last updated: January 2025.",
        "Acte de cession de droits de propriété intellectuelle. Le cédant transfère à titre exclusif les droits sur {topic} au cessionnaire.",
        "Legal notice: {topic} Corporation registered under company number {n}. Registered office: ... VAT number: FR{n}. Capital: €10,000.",
    ],
    "Medical / Health": [
        "Patient: ... Age: {n}. Diagnosis: {topic}. Treatment plan: prescribed medication for 10 days, follow-up in 2 weeks. Physician: Dr. ...",
        "Compte rendu médical — Patient : ... Motif de consultation : {topic}. Examen clinique : normal. Prescription : ... Prochain RDV : ...",
        "Clinical guidelines for {topic} management. Recommended treatment: first-line therapy includes... Second-line options: ... Contraindications...",
        "Medical report: imaging results for {topic}. Findings: no significant abnormality detected. Recommendation: routine follow-up in 6 months.",
        "Ordonnance médicale — Médicament : {topic}. Posologie : 1 comprimé matin et soir pendant 7 jours. Ne pas dépasser la dose prescrite.",
        "Discharge summary: Patient admitted for {topic}. Hospital stay: {n} days. Condition on discharge: stable. Follow-up instructions provided.",
        "Health screening report: {topic} indicators are within normal range. BMI: 22.5. Blood pressure: 120/80. Cholesterol: 180 mg/dL.",
        "Rapport d'analyse de laboratoire — Examens réalisés : NFS, bilan hépatique, {topic}. Résultats : dans les normes. Commentaires : ...",
        "Vaccination record: {topic} vaccine administered. Date: 10/05/2025. Batch: LOT-{n}. Next dose: in 6 months. No adverse reactions noted.",
        "Protocol for {topic} clinical trial. Inclusion criteria: adults 18-65, diagnosis confirmed. Exclusion criteria: pregnancy, severe renal failure.",
    ],
    "Technical / Engineering": [
        "System architecture for {topic}: microservices deployed on Kubernetes. Components: API gateway, authentication service, database cluster.",
        "Technical specification — {topic} module. Input: JSON payload with fields... Output: REST API response. Error codes: 400, 401, 500.",
        "Algorithm design for {topic}: time complexity O(n log n), space complexity O(n). Implementation in Python 3.10 using NumPy and Pandas.",
        "Réseau {topic} : protocole TCP/IP, topologie en étoile, débit maximal 1 Gbps. Configuration du routeur : adresse IP, masque, passerelle.",
        "Software design document for {topic}. Architecture: MVC pattern. Database: PostgreSQL. Frontend: React. Backend: FastAPI. Deployment: Docker.",
        "Circuit design for {topic}: input voltage 5V, current 2A. Components: resistors, capacitors, MOSFET transistors. PCB layout attached.",
        "API documentation for {topic}. Endpoint: POST /api/v1/{topic}. Request body: JSON object with required fields. Response: 200 OK with JSON payload.",
        "Code review checklist for {topic} module. Security: SQL injection prevention, input validation. Performance: caching, pagination implemented.",
        "Deployment guide for {topic} application. Prerequisites: Docker, Python 3.10+. Steps: clone repo, configure .env, run docker-compose up.",
        "Performance benchmark for {topic} system. Throughput: 1000 req/s. Latency p95: 45ms. Memory: 512MB. CPU: 15% under load.",
    ],
}

TOPICS = [
    "machine learning", "data structures", "neural networks", "algorithms",
    "database systems", "computer vision", "natural language processing",
    "cloud computing", "cybersecurity", "software engineering", "IoT systems",
    "signal processing", "distributed systems", "network protocols",
    "operating systems", "project management", "financial analysis",
    "supply chain", "human resources", "marketing strategy",
]


def generate_dataset() -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for label, templates in TEMPLATES.items():
        for i in range(SAMPLES_PER_CLASS):
            tmpl  = templates[i % len(templates)]
            topic = random.choice(TOPICS)
            n     = random.randint(1, 99)
            text  = tmpl.format(topic=topic, n=n)
            # Augmentation : concaténer 2-3 phrases du même domaine
            extra = random.choice(templates)
            text += " " + extra.format(topic=random.choice(TOPICS), n=random.randint(1,99))
            texts.append(text)
            labels.append(label)
    return texts, labels


# ── Modèle (identique à classifier_tool.py) ───────────────────────────────────

class SimpleDocumentClassifier(nn.Module):
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

    def forward(self, x):
        return self.net(x)


# ── Entraînement ──────────────────────────────────────────────────────────────

def train():
    print("\n" + "="*60)
    print("  SmartDoc AI — Entraînement du classificateur PyTorch")
    print("="*60)

    # 1. Dataset
    print("\n[1/5] Génération du dataset synthétique...")
    texts, labels = generate_dataset()
    label_set     = sorted(set(labels))
    label_to_id   = {l: i for i, l in enumerate(label_set)}
    id_to_label   = {i: l for l, i in label_to_id.items()}
    y             = [label_to_id[l] for l in labels]
    print(f"      {len(texts)} échantillons · {len(label_set)} catégories")
    for lbl in label_set:
        print(f"        • {lbl}: {labels.count(lbl)} samples")

    # 2. TF-IDF
    print("\n[2/5] Vectorisation TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    print(f"      Matrice features : {X.shape}")

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train : {len(X_train)} | Test : {len(X_test)}")

    # 4. Entraînement
    print(f"\n[3/5] Entraînement ({EPOCHS} epochs)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device : {device}")

    model = SimpleDocumentClassifier(X.shape[1], len(label_set)).to(device)

    X_tr = torch.tensor(X_train).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{EPOCHS} — loss: {avg:.4f}")

    # 5. Évaluation
    print("\n[4/5] Évaluation sur le jeu de test...")
    model.eval()
    X_te = torch.tensor(X_test).to(device)
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1).cpu().numpy()

    acc      = accuracy_score(y_test, preds)
    report   = classification_report(y_test, preds,
                                     target_names=label_set, output_dict=True)
    cm       = confusion_matrix(y_test, preds)

    print(f"\n      [OK] Accuracy : {acc*100:.2f}%")
    print("\n      Classification Report :")
    print(classification_report(y_test, preds, target_names=label_set))

    # Matrice de confusion
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_set)))
    ax.set_yticks(range(len(label_set)))
    short = [l.split("/")[0].strip()[:12] for l in label_set]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Prédit", fontsize=11)
    ax.set_ylabel("Réel",   fontsize=11)
    ax.set_title(f"Matrice de Confusion — SmartDoc AI\nAccuracy : {acc*100:.2f}%", fontsize=13)
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=9)
    plt.tight_layout()
    cm_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Matrice sauvegardée : {cm_path}")

    # Sauvegarde métriques JSON
    metrics = {
        "accuracy"        : round(acc, 4),
        "accuracy_pct"    : round(acc * 100, 2),
        "epochs"          : EPOCHS,
        "best_train_loss" : round(best_loss, 4),
        "num_classes"     : len(label_set),
        "num_samples"     : len(texts),
        "tfidf_features"  : TFIDF_MAX_FEATURES,
        "per_class"       : {
            lbl: {
                "precision": round(report[lbl]["precision"], 3),
                "recall"   : round(report[lbl]["recall"],    3),
                "f1-score" : round(report[lbl]["f1-score"],  3),
                "support"  : report[lbl]["support"],
            }
            for lbl in label_set
        },
    }
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"      Métriques sauvegardées : {metrics_path}")

    # 6. Sauvegarde modèle
    print("\n[5/5] Sauvegarde des artefacts...")
    torch.save(model.state_dict(), MODELS_DIR / "document_classifier.pth")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump(id_to_label, MODELS_DIR / "id_to_label.pkl")
    print(f"      [OK] document_classifier.pth")
    print(f"      [OK] vectorizer.pkl")
    print(f"      [OK] id_to_label.pkl")

    print("\n" + "="*60)
    print(f"  ENTRAÎNEMENT TERMINÉ — Accuracy : {acc*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    train()
