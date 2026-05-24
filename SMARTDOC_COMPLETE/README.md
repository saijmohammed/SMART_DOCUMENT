# SmartDoc AI — Projet Intégré S8 UIR

**Système multi-agents d'analyse intelligente de documents PDF**

Encadrant : Prof. Hakim Hafidi — Programme AI & Big Data — UIR S8

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du système](#2-architecture-du-système)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation et configuration](#4-installation-et-configuration)
5. [Lancement de l'application](#5-lancement-de-lapplication)
6. [Entraînement du modèle Deep Learning](#6-entraînement-du-modèle-deep-learning)
7. [Détail des agents](#7-détail-des-agents)
8. [Détail des outils (Tools)](#8-détail-des-outils-tools)
9. [Modèle PyTorch — Classification de documents](#9-modèle-pytorch--classification-de-documents)
10. [Interface utilisateur Streamlit](#10-interface-utilisateur-streamlit)
11. [Génération de rapports](#11-génération-de-rapports)
12. [Fournisseurs LLM — Chaîne de priorité](#12-fournisseurs-llm--chaîne-de-priorité)
13. [Pipeline CLI (main.py)](#13-pipeline-cli-mainpy)
14. [Logging et traçabilité](#14-logging-et-traçabilité)
15. [Variables d'environnement](#15-variables-denvironnement)
16. [Dépendances](#16-dépendances)
17. [Résultats du modèle](#17-résultats-du-modèle)
18. [Conformité au cahier des charges UIR](#18-conformité-au-cahier-des-charges-uir)

---

## 1. Vue d'ensemble

SmartDoc AI est une application web d'analyse de documents PDF construite selon une **architecture multi-agents**. L'utilisateur charge un PDF ; l'orchestrateur coordonne automatiquement 7 agents spécialistes qui extraient, classifient, résument, expliquent, traduisent le contenu, puis génèrent un rapport structuré (PDF ou Word).

Un **checkpoint Human-in-the-Loop (HITL)** oblige l'opérateur humain à valider les résultats avant la génération du rapport final.

**Fonctionnalités principales :**

- Upload et extraction de texte PDF (PyMuPDF)
- Classification automatique du type de document (modèle PyTorch + TF-IDF)
- Résumé intelligent (Gemini API ou extractif TF-IDF)
- Explication détaillée du contenu (Groq / Gemini / local)
- Traduction multilingue FR / EN / AR (DeepTranslator + Gemini)
- Chatbot IA sur le document (Groq Llama 3.3 / Gemini / local)
- Génération de rapports professionnels PDF et Word (ReportLab + python-docx)
- Interface ChatGPT-style responsive (Streamlit + CSS personnalisé)
- Validation humaine obligatoire avant export (HITL)
- Logging JSON horodaté de toutes les actions

---

## 2. Architecture du système

```
┌─────────────────────────────────────────────────────────────────┐
│                        Interface Streamlit                       │
│                           app.py                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ upload PDF + paramètres
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SmartDocOrchestrator                          │
│              src/agents/orchestrator.py                         │
│                                                                  │
│  1. ExtractorAgent ──► ExtractorTool (PyMuPDF)                  │
│  2. ClassifierAgent ──► ClassifierTool (PyTorch + TF-IDF)       │
│  3. SummarizerAgent ──► SummarizerTool (Gemini / TF-IDF)        │
│  4. ExplainerAgent ──► NlpTool (Gemini / local)                 │
│  5. TranslatorAgent ──► TranslatorTool (DeepTranslator)         │
│                                                                  │
│  ──────── CHECKPOINT HITL (validation humaine) ────────         │
│                                                                  │
│  6. ReportAgent ──► ReportTool (ReportLab + python-docx)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        Rapport PDF               Rapport Word
      (ReportLab)               (python-docx)

     ChatAgent ──► ChatTool ──► Groq API → Gemini → Local
```

**Flux de données :**

```
PDF → [Extraction] → texte brut
     → [Classification] → catégorie + confiance (PyTorch)
     → [Résumé] → résumé + points clés
     → [Explication] → analyse détaillée
     → [Traduction] → texte traduit (optionnel)
     → [HITL] → validation opérateur
     → [Rapport] → PDF / Word téléchargeable
```

---

## 3. Structure du projet

```
SMARTDOC_COMPLETE/
│
├── app.py                      # Interface Streamlit principale
├── main.py                     # CLI pipeline (usage: python main.py doc.pdf)
├── train_model.py              # Script d'entraînement PyTorch
├── requirements.txt            # Dépendances Python
├── .env                        # Variables d'environnement (non versionné)
├── .env.example                # Modèle de configuration
├── .gitignore
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py     # Coordinateur multi-agents + HITL
│   │   ├── extractor_agent.py  # Agent d'extraction PDF
│   │   ├── classifier_agent.py # Agent de classification (DL)
│   │   ├── summarizer_agent.py # Agent de résumé
│   │   ├── explainer_agent.py  # Agent d'explication
│   │   ├── translator_agent.py # Agent de traduction
│   │   ├── report_agent.py     # Agent de génération de rapport
│   │   └── chat_agent.py       # Agent chatbot
│   │
│   ├── tools/
│   │   ├── extractor_tool.py   # Extraction texte PDF (PyMuPDF)
│   │   ├── classifier_tool.py  # Modèle PyTorch + fallback heuristique
│   │   ├── summarizer_tool.py  # Résumé Gemini / extractif TF-IDF
│   │   ├── nlp_tool.py         # Explication Gemini / local
│   │   ├── translator_tool.py  # Traduction DeepTranslator / Gemini
│   │   ├── report_tool.py      # Génération PDF (ReportLab) + Word (python-docx)
│   │   ├── chat_tool.py        # Chatbot Groq / Gemini / local
│   │   └── logger_tool.py      # Logging JSON horodaté
│   │
│   └── utils/
│       ├── config.py           # Centralisation config + chemins + clés API
│       ├── pdf_reader.py       # Lecture PDF page par page (PyMuPDF)
│       └── text_cleaner.py     # Nettoyage, chunking, troncature texte
│
├── models/
│   ├── document_classifier.pth # Poids du réseau PyTorch (entraîné)
│   ├── vectorizer.pkl          # TF-IDF vectorizer (scikit-learn)
│   ├── id_to_label.pkl         # Mapping indice → catégorie
│   ├── evaluation_metrics.json # Métriques d'évaluation (accuracy, F1...)
│   └── confusion_matrix.png    # Matrice de confusion (visualisation)
│
├── reports/                    # Rapports générés (PDF + Word)
├── uploads/                    # PDFs uploadés par l'utilisateur
├── data/                       # Données (réservé)
└── logs/
    └── actions.json            # Log JSON de toutes les actions
```

---

## 4. Installation et configuration

### Prérequis

- Python 3.10 ou supérieur
- pip

### Étapes

**1. Cloner ou copier le projet**

```bash
cd C:\Users\pc\Desktop\SMARTDOC_COMPLETE
```

**2. Créer l'environnement virtuel**

```bash
python -m venv .venv
```

**3. Activer l'environnement virtuel**

Windows :
```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS :
```bash
source .venv/bin/activate
```

**4. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**5. Configurer les clés API**

Créer un fichier `.env` à la racine du projet (ou copier `.env.example`) :

```env
APP_NAME=SmartDoc AI
GEMINI_API_KEY=votre_cle_gemini_ici
GEMINI_MODEL=gemini-2.0-flash
GROQ_API_KEY=votre_cle_groq_ici
GROQ_MODEL=llama-3.3-70b-versatile
```

- **Groq (recommandé, gratuit)** : créer un compte sur [console.groq.com](https://console.groq.com) → API Keys → Create Key
- **Gemini (optionnel, fallback)** : créer une clé sur [aistudio.google.com](https://aistudio.google.com/app/apikey)

> L'application fonctionne également sans clés API grâce aux fallbacks locaux, mais avec une qualité de réponse réduite.

---

## 5. Lancement de l'application

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

L'interface s'ouvre automatiquement sur `http://localhost:8501`.

**Workflow utilisateur :**

1. Charger un fichier PDF dans la barre latérale gauche
2. Cocher les options souhaitées (résumé, explication, traduction)
3. Cliquer sur **"Analyser le document"**
4. Visualiser les résultats de chaque agent (classification, résumé, explication)
5. Valider les résultats via le **checkpoint HITL** (bouton "Valider et générer le rapport")
6. Télécharger le rapport final en PDF ou Word
7. Utiliser l'onglet **Chat** pour poser des questions sur le document

---

## 6. Entraînement du modèle Deep Learning

Le modèle PyTorch doit être entraîné avant le premier lancement (les fichiers sont déjà présents dans `models/` si vous avez cloné le projet complet).

Pour ré-entraîner :

```powershell
.venv\Scripts\python.exe train_model.py
```

**Ce que fait le script :**

1. **Génère un dataset synthétique** : 960 exemples équilibrés (120 par catégorie × 8 catégories) en utilisant des templates de phrases réalistes avec substitutions aléatoires de topics
2. **Vectorisation TF-IDF** : matrice features (960, 4000) avec bigrammes et sublinear_tf
3. **Entraînement** : 40 epochs, Adam optimizer (lr=1e-3, weight_decay=1e-4), StepLR scheduler, device CPU/GPU auto-détecté
4. **Évaluation** : accuracy, rapport de classification, matrice de confusion
5. **Sauvegarde** dans `models/` : 5 fichiers (voir section 9)

Durée estimée : 2-5 minutes sur CPU.

---

## 7. Détail des agents

### Orchestrateur — `src/agents/orchestrator.py`

Coordonne l'ensemble du pipeline. Expose deux méthodes publiques :

- `analyze(pdf_path, ...)` — exécute les étapes 1 à 5 et retourne un dict complet avec `status: "success"`
- `generate_report_after_human_validation(result, validator_note)` — checkpoint HITL, puis génère le rapport

Paramètres optionnels de `analyze()` :

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `do_summary` | bool | True | Activer le résumé |
| `do_explanation` | bool | True | Activer l'explication |
| `do_translation` | bool | False | Activer la traduction |
| `target_language` | str | "fr" | Code langue cible |

### ExtractorAgent — `src/agents/extractor_agent.py`

Délègue à `extractor_tool.extract_pdf_text()`. Lit le PDF page par page, nettoie les artefacts d'extraction, retourne le texte complet + métadonnées (nom, pages, mots, caractères).

### ClassifierAgent — `src/agents/classifier_agent.py`

Délègue à `classifier_tool.classify_document()`. Charge le modèle PyTorch entraîné, vectorise le texte avec TF-IDF, retourne la catégorie prédite + score de confiance + scores pour toutes les classes.

### SummarizerAgent — `src/agents/summarizer_agent.py`

Délègue à `summarizer_tool.summarize_text()`. Priorité : Gemini API → résumé extractif TF-IDF. Retourne un résumé court + liste de points clés.

### ExplainerAgent — `src/agents/explainer_agent.py`

Délègue à `nlp_tool.explain_document()`. Génère une explication contextuelle selon le type de document (académique, technique, légal...). Priorité : Gemini → local.

### TranslatorAgent — `src/agents/translator_agent.py`

Délègue à `translator_tool.translate_text()`. Traduction FR ↔ EN ↔ AR. Priorité : DeepTranslator (googletrans) → Gemini → substitution locale minimale.

### ReportAgent — `src/agents/report_agent.py`

Délègue à `report_tool`. Génère un rapport PDF professionnel (page de couverture, sections colorées, tableau de métadonnées, liste à puces, matrice de tags) et un rapport Word équivalent.

### ChatAgent — `src/agents/chat_agent.py`

Délègue à `chat_tool.answer_question()`. Maintient le contexte de la conversation. Priorité LLM : Groq → Gemini → réponse locale intelligente.

---

## 8. Détail des outils (Tools)

### `extractor_tool.py`

```
Entrée  : pdf_path (str)
Sortie  : { status, file_name, page_count, text, char_count, word_count, pages[] }
```

Utilise **PyMuPDF** (`fitz`) pour extraire le texte page par page. Détecte les PDFs scannés (pas de texte lisible) et retourne une erreur explicite. Ne lève jamais d'exception.

### `classifier_tool.py`

```
Entrée  : text (str)
Sortie  : { status, category, confidence, mode, all_scores{} }
```

Deux modes :
- **`pytorch_model`** : charge `vectorizer.pkl` → transforme le texte → inference `SimpleDocumentClassifier` → softmax → catégorie + confiance
- **`heuristic_fallback`** : comptage de mots-clés si les fichiers modèle sont absents

### `summarizer_tool.py`

```
Entrée  : text (str), max_sentences (int, défaut 8)
Sortie  : { status, summary, key_points[], mode, title }
```

Mode **Gemini** : prompt structuré demandant titre + résumé + 5 points clés en Markdown.
Mode **extractif** : score TF-IDF par phrase (fréquence × rareté des termes), sélection des top-N phrases, extraction des points clés par fréquence de termes significatifs.

### `nlp_tool.py`

```
Entrée  : text (str), doc_type (str)
Sortie  : { status, explanation, structure, tech_terms[], questions[], mode }
```

Génère une explication adaptée au type de document. Extrait les termes techniques (40+ noms connus + patterns CamelCase). Génère des questions de compréhension.

### `translator_tool.py`

```
Entrée  : text (str), target_lang (str: "fr"|"en"|"ar")
Sortie  : { status, translated_text, source_lang, target_lang, mode }
```

Priorité : **DeepTranslator** (googletrans backend) → Gemini (si clé présente) → fallback local minimal.

### `report_tool.py`

Génère deux formats de rapport :

**PDF (ReportLab) :**
- Page de couverture sombre avec dégradé (`#0f172a`)
- En-têtes de section colorés (`#1d4ed8` bleu)
- Tableau de métadonnées du document
- Résumé formaté avec puces colorées
- Section explication structurée
- Traduction si disponible
- Pied de page avec numérotation et timestamp

**Word (python-docx) :**
- Structure identique au PDF
- Cellules de tableau avec fond coloré (XML `w:shd`)
- Styles Heading 1/2, Normal, ListBullet
- Bordures de séparation XML

Palette de couleurs :

| Constante | Hex | Usage |
|-----------|-----|-------|
| `C_DARK` | `#0f172a` | Fond couverture, texte sombre |
| `C_BLUE` | `#1d4ed8` | En-têtes de section |
| `C_GREEN` | `#059669` | Indicateurs de succès |
| `C_VIOLET` | `#7c3aed` | Tags, accents |

### `chat_tool.py`

```
Entrée  : question (str), doc_text (str), history (list)
Sortie  : { status, answer, mode, sources[] }
```

**Chaîne LLM :**
1. **Groq** (`llama-3.3-70b-versatile`) — 14 400 requêtes/jour gratuites
2. **Gemini** (`gemini-2.0-flash`) — fallback si Groq échoue
3. **Réponse locale intelligente** — fallback final sans API

Le prompt envoyé aux LLMs inclut :
- Instruction de formatage Markdown strict (`##` titres, `-` listes, `**bold**`)
- Contexte du document (premiers 3000 caractères)
- Historique des 4 derniers échanges
- La question posée

Détection automatique du type de question : structure, conclusion, outils, objectif, auteur, méthode, exigences, général.

### `logger_tool.py`

```python
log_action(agent, action, status, data)
```

Append dans `logs/actions.json` au format :
```json
{
  "timestamp": "2026-05-24T14:08:51.123Z",
  "agent": "ClassifierAgent",
  "action": "classify_document",
  "status": "success",
  "data": { ... }
}
```

---

## 9. Modèle PyTorch — Classification de documents

### Architecture du réseau

```
Entrée : vecteur TF-IDF (4000 dimensions)
   │
   ├── Linear(4000 → 512)
   ├── BatchNorm1d(512)
   ├── ReLU
   ├── Dropout(0.3)
   │
   ├── Linear(512 → 256)
   ├── BatchNorm1d(256)
   ├── ReLU
   ├── Dropout(0.2)
   │
   ├── Linear(256 → 128)
   ├── ReLU
   │
   └── Linear(128 → 8)  ← logits pour 8 catégories
```

### Hyperparamètres d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Epochs | 40 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Optimizer | Adam |
| Scheduler | StepLR (step=10, gamma=0.5) |
| Features TF-IDF | 4 000 |
| N-grams | (1, 2) — unigrammes + bigrammes |
| sublinear_tf | True |

### Dataset synthétique

960 exemples équilibrés — 120 par catégorie :

| Catégorie | Exemples de contenu |
|-----------|---------------------|
| Course / Academic | Cours, théorèmes, exercices, syllabus |
| Research Paper | Abstract, DOI, methodology, findings |
| Report | Rapport, objectifs, recommandations |
| Business / Invoice | Factures, paiements, clients, TVA |
| Administrative | Attestations, formulaires, arrêtés |
| Legal / Contract | Contrats, clauses, NDA, baux |
| Medical / Health | Ordonnances, diagnostics, protocoles |
| Technical / Engineering | Architecture, API, spécifications |

### Résultats

| Métrique | Valeur |
|----------|--------|
| Accuracy (test set) | **100.00%** |
| Precision macro | 1.000 |
| Recall macro | 1.000 |
| F1-score macro | 1.000 |
| Epochs | 40 |
| Best train loss | 0.0003 |
| Test samples | 192 (20% du dataset) |

Toutes les classes atteignent precision = recall = f1 = 1.00.

---

## 10. Interface utilisateur Streamlit

### Onglets principaux

| Onglet | Contenu |
|--------|---------|
| Analyse | Upload PDF, options, pipeline, résultats agents |
| Chat | Chatbot IA sur le document (style ChatGPT) |
| Rapports | Historique des rapports générés, téléchargement |
| À propos | Informations projet, version, crédits |

### Design chat

L'interface chat reproduit le style des assistants modernes (ChatGPT / Claude) :

- Avatar bot : carré dégradé bleu-violet avec symbole `✦`
- Avatar utilisateur : cercle gris `👤`
- Messages bot : texte sans bulle, fond transparent, largeur pleine
- Messages utilisateur : bulle arrondie, alignée à droite
- Rendu Markdown : `##` → titres colorés, `-` → listes, `**text**` → gras
- Badge de mode LLM en temps réel (`✦ Groq — Llama 3.3` / `Gemini` / `Local`)

### Mode Chip (barre latérale)

Affiche le fournisseur LLM actif :

- `✦ Groq — Llama 3.3` (fond bleu) — si `GROQ_API_KEY` définie
- `◆ Gemini 2.0 Flash` (fond violet) — si `GEMINI_API_KEY` définie
- `⬡ Mode Local` (fond gris) — sans clés API

---

## 11. Génération de rapports

### Types de rapports

| Type | Format | Contenu |
|------|--------|---------|
| Summary | PDF / Word | Résumé + points clés + métadonnées |
| General | PDF / Word | Rapport complet (résumé + explication + traduction + classification) |
| Translation | PDF / Word | Texte traduit uniquement |

### Nommage des fichiers

```
smartdoc_{type}_{YYYYMMDD}_{HHMMSS}.pdf
smartdoc_{type}_{YYYYMMDD}_{HHMMSS}.docx
```

Exemples : `smartdoc_summary_20260524_165036.pdf`, `smartdoc_general_20260524_140851.docx`

### Historique

Chaque rapport généré est enregistré dans `reports/history.json` avec timestamp, type, chemin, et métadonnées du document.

---

## 12. Fournisseurs LLM — Chaîne de priorité

```
Question posée
      │
      ▼
 Groq disponible ? ──► OUI ──► llama-3.3-70b-versatile (14 400 req/jour gratuit)
      │
      NON
      │
      ▼
 Gemini disponible ? ──► OUI ──► gemini-2.0-flash (quota limité)
      │
      NON
      │
      ▼
 Réponse locale intelligente (analyse TF-IDF + détection type question)
```

**Avantages Groq :**
- Gratuit jusqu'à 14 400 requêtes/jour
- Latence très faible (~500ms)
- Modèle Llama 3.3 70B de haute qualité
- Pas de quota d'expiration

**Avantages Gemini :**
- Modèle Google multimodal
- Fallback fiable si Groq indisponible

**Réponse locale :**
- Fonctionne sans internet
- Basée sur l'analyse du texte du document
- Moins performante mais toujours utile

---

## 13. Pipeline CLI (`main.py`)

En plus de l'interface web, un pipeline en ligne de commande est disponible :

```bash
python main.py chemin/vers/document.pdf
python main.py chemin/vers/document.pdf --translate --lang en
python main.py chemin/vers/document.pdf --no-explain
```

Options :

| Option | Description |
|--------|-------------|
| `pdf` | Chemin vers le fichier PDF (obligatoire) |
| `--no-summary` | Désactiver le résumé |
| `--no-explain` | Désactiver l'explication |
| `--translate` | Activer la traduction |
| `--lang` | Langue cible : `fr`, `en`, `ar` (défaut: `fr`) |
| `--output` | Répertoire de sortie (défaut: `reports/`) |

---

## 14. Logging et traçabilité

Toutes les actions de tous les agents sont enregistrées dans `logs/actions.json`.

Structure d'un log :

```json
{
  "timestamp": "2026-05-24T14:08:51.123456+00:00",
  "agent": "ClassifierAgent",
  "action": "classify_document",
  "status": "success",
  "data": {
    "category": "Course / Academic",
    "confidence": 0.9871,
    "mode": "pytorch_model"
  }
}
```

Agents loggés : `ExtractorAgent`, `ClassifierAgent`, `SummarizerAgent`, `ExplainerAgent`, `TranslatorAgent`, `ReportAgent`, `ChatAgent`, `Orchestrator`.

---

## 15. Variables d'environnement

| Variable | Obligatoire | Défaut | Description |
|----------|-------------|--------|-------------|
| `APP_NAME` | Non | `SmartDoc AI` | Nom de l'application |
| `GEMINI_API_KEY` | Non | `` | Clé API Google Gemini |
| `GEMINI_MODEL` | Non | `gemini-2.0-flash` | Modèle Gemini à utiliser |
| `GROQ_API_KEY` | Non | `` | Clé API Groq |
| `GROQ_MODEL` | Non | `llama-3.3-70b-versatile` | Modèle Groq à utiliser |

> Sans aucune clé API, l'application fonctionne en mode entièrement local (classification PyTorch, résumé extractif, réponses locales).

---

## 16. Dépendances

| Package | Version | Rôle |
|---------|---------|------|
| `streamlit` | >=1.35.0 | Interface web |
| `PyMuPDF` | >=1.24.0 | Extraction texte PDF |
| `reportlab` | >=4.2.0 | Génération PDF |
| `torch` | >=2.2.0 | Réseau de neurones PyTorch |
| `numpy` | >=1.26.0 | Calcul numérique |
| `pandas` | >=2.2.0 | Manipulation de données |
| `scikit-learn` | >=1.4.0 | TF-IDF vectorizer, métriques |
| `matplotlib` | >=3.8.0 | Matrice de confusion |
| `python-docx` | >=1.1.0 | Génération Word |
| `google-genai` | >=1.0.0 | Google Gemini API (nouveau SDK) |
| `groq` | >=0.9.0 | Groq API (Llama 3.3) |
| `python-dotenv` | >=1.0.0 | Chargement `.env` |
| `deep-translator` | >=1.11.0 | Traduction multilingue |
| `joblib` | >=1.4.0 | Sauvegarde/chargement modèles |

Installation :
```bash
pip install -r requirements.txt
```

---

## 17. Résultats du modèle

### Métriques d'évaluation (test set — 192 échantillons)

| Catégorie | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Administrative | 1.000 | 1.000 | 1.000 | 24 |
| Business / Invoice | 1.000 | 1.000 | 1.000 | 24 |
| Course / Academic | 1.000 | 1.000 | 1.000 | 24 |
| Legal / Contract | 1.000 | 1.000 | 1.000 | 24 |
| Medical / Health | 1.000 | 1.000 | 1.000 | 24 |
| Report | 1.000 | 1.000 | 1.000 | 24 |
| Research Paper | 1.000 | 1.000 | 1.000 | 24 |
| Technical / Engineering | 1.000 | 1.000 | 1.000 | 24 |
| **Accuracy globale** | | | **1.000** | **192** |

**Best training loss :** 0.0003 (epoch 40)

La matrice de confusion est disponible dans `models/confusion_matrix.png`.

---

## 18. Conformité au cahier des charges UIR

| Exigence | Statut | Détail |
|----------|--------|--------|
| Architecture multi-agents | ✅ | 7 agents spécialistes + 1 orchestrateur |
| Modèle Deep Learning fonctionnel | ✅ | PyTorch FC (TF-IDF → 4 couches → 8 classes), accuracy 100% |
| Human-in-the-Loop (HITL) | ✅ | Checkpoint obligatoire avant génération rapport |
| Gestion d'erreurs robuste | ✅ | Aucun agent ne lève d'exception, fallbacks à chaque niveau |
| Logging horodaté | ✅ | JSON dans `logs/actions.json` |
| Interface utilisateur | ✅ | Streamlit multi-onglets, design ChatGPT-style |
| Génération de rapports | ✅ | PDF (ReportLab) + Word (python-docx), professionnels |
| Pipeline CLI | ✅ | `main.py` avec argparse |
| LLM intégré | ✅ | Groq (Llama 3.3) + Gemini (gemini-2.0-flash) |
| Traduction multilingue | ✅ | FR / EN / AR |
| Chatbot document | ✅ | IA contextuelle sur le contenu du PDF |

---

## Informations projet

**Projet :** SmartDoc AI — Système Multi-Agents d'Analyse de Documents  
**Cours :** Integrated Project — S8 AI & Big Data  
**Université :** UIR (Université Internationale de Rabat)  
**Encadrant :** Prof. Hakim Hafidi  
**Version :** 4.0.0
