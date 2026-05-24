"""
app.py — SmartDoc AI v4.0
Plateforme d'analyse PDF multi-agents.

Pages :
  home          — Upload + lancement pipeline + dashboard résultats
  summary       — Résumé détaillé structuré + export PDF/Word
  explanation   — Explication + termes + vidéos YouTube + questions
  translation   — Traduction interactive (12 langues) + export
  general_report— Rapport général HITL + export
  history       — Historique des 30 dernières analyses
"""
import json
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.agents.orchestrator    import SmartDocOrchestrator
from src.agents              import chat_agent
from src.tools.logger_tool      import clear_logs
from src.tools.translator_tool  import translate_text, ALL_LANGUAGES, get_lang_label
from src.utils.config           import APP_VERSION, UPLOADS_DIR
from src.tools.report_tool      import (
    generate_general_docx_report,
    generate_general_pdf_report,
    generate_summary_docx_report,
    generate_summary_pdf_report,
    generate_translation_docx_report,
    generate_translation_pdf_report,
)

# =============================================================================
# CONFIG STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_PATH = ROOT / "reports" / "history.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CSS — DARK EDITORIAL DESIGN
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#050814; --bg2:#070c1b; --surface:#0d1327; --surface2:#111a31; --surface3:#17213b;
  --border:rgba(99,102,241,.22); --border2:rgba(56,189,248,.38); --border3:rgba(16,185,129,.30);
  --text:#e5eefc; --muted:#9aa9c1; --dim:#60708a;
  --blue:#3b82f6; --cyan:#06b6d4; --green:#10b981; --amber:#f59e0b;
  --rose:#f43f5e; --violet:#8b5cf6; --r:18px;
}

html,body,[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 10% 2%,rgba(59,130,246,.13),transparent 30%),
    radial-gradient(circle at 85% 8%,rgba(139,92,246,.11),transparent 32%),
    radial-gradient(circle at 48% 92%,rgba(6,182,212,.06),transparent 38%),
    linear-gradient(145deg,#050814 0%,#060b18 60%,#040710 100%) !important;
  color:var(--text)!important; font-family:'DM Sans',sans-serif!important;
}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#07101f,#050814)!important;
  border-right:1px solid var(--border)!important;
}
.block-container{padding:1.2rem 2rem!important;max-width:100%!important}
#MainMenu,footer,header{visibility:hidden}
h1,h2,h3{font-family:'Syne',sans-serif!important}
label,.stMarkdown,p{color:var(--text)}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#2563eb,#4f46e5)!important;color:#fff!important;
  border:none!important;border-radius:14px!important;font-family:'Syne',sans-serif!important;
  font-weight:800!important;letter-spacing:.03em!important;
  box-shadow:0 8px 22px rgba(59,130,246,.22)!important;transition:all .18s!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 14px 36px rgba(59,130,246,.36)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#1d4ed8,#6d28d9)!important;box-shadow:0 10px 30px rgba(99,102,241,.30)!important}
.stDownloadButton>button{
  background:linear-gradient(135deg,#0f766e,#0891b2)!important;color:#fff!important;
  border:none!important;border-radius:14px!important;font-family:'Syne',sans-serif!important;
  font-weight:800!important;box-shadow:0 8px 22px rgba(6,182,212,.22)!important;transition:all .18s!important;
}
.stDownloadButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 12px 32px rgba(6,182,212,.34)!important}
[data-testid="stFileUploader"]{
  background:rgba(17,24,39,.75)!important;border:2px dashed rgba(99,102,241,.50)!important;
  border-radius:22px!important;padding:.9rem!important;
}
.stTextArea textarea,.stTextInput input{
  background:var(--surface2)!important;border:1px solid var(--border2)!important;
  border-radius:13px!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;
}
.stSelectbox div[data-baseweb="select"]>div{
  background:var(--surface2)!important;border:1px solid var(--border2)!important;
  border-radius:13px!important;color:var(--text)!important;
}

/* Hero */
.hero{
  background:linear-gradient(135deg,rgba(13,19,39,.96),rgba(17,26,52,.93));
  border:1px solid var(--border2);border-radius:28px;padding:2.2rem 2.6rem;
  margin-bottom:1.5rem;position:relative;overflow:hidden;
}
.hero::before{content:"";position:absolute;width:480px;height:480px;right:-180px;top:-200px;
  background:radial-gradient(circle,rgba(56,189,248,.15),transparent 72%);pointer-events:none}
.hero::after{content:"";position:absolute;width:340px;height:340px;left:22%;bottom:-195px;
  background:radial-gradient(circle,rgba(139,92,246,.12),transparent 70%);pointer-events:none}
.hero-badge{
  display:inline-block;padding:5px 15px;border-radius:999px;
  background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.30);
  color:#93c5fd;text-transform:uppercase;letter-spacing:.14em;font-weight:900;font-size:.68rem;margin-bottom:1rem;
}
.hero-title{
  font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:900;line-height:.95;
  background:linear-gradient(135deg,#f8fafc,#93c5fd 55%,#c4b5fd);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.9rem;
}
.hero-sub{color:var(--muted);line-height:1.85;max-width:840px;font-size:.97rem}
.hero-version{position:absolute;top:1.2rem;right:1.6rem;color:var(--dim);font-family:'JetBrains Mono',monospace;font-size:.7rem}

/* Section title */
.section-title{
  font-family:'Syne',sans-serif;font-size:1.12rem;font-weight:900;
  margin:1.4rem 0 .85rem;display:flex;align-items:center;gap:10px;
}
.section-title::after{content:"";height:1px;flex:1;background:linear-gradient(90deg,var(--border2),transparent)}

/* Cards */
.card{
  background:rgba(13,19,39,.88);border:1px solid var(--border);border-radius:var(--r);
  padding:1.3rem 1.4rem;margin-bottom:1rem;box-shadow:0 16px 44px rgba(0,0,0,.16);
}
.card-title{font-family:'Syne',sans-serif;font-weight:900;font-size:1rem;margin-bottom:.45rem}
.card-text{color:var(--muted);font-size:.86rem;line-height:1.78}
.upload-box{
  background:linear-gradient(135deg,rgba(59,130,246,.09),rgba(139,92,246,.07));
  border:1px solid var(--border);border-radius:var(--r);padding:1.45rem;
  margin-bottom:1rem;box-shadow:0 16px 44px rgba(0,0,0,.14);
}
.module-card{
  background:rgba(13,19,39,.88);border:1px solid var(--border);border-radius:var(--r);
  padding:1.4rem;min-height:170px;transition:border-color .2s,transform .2s,box-shadow .2s;
  box-shadow:0 10px 28px rgba(0,0,0,.14);
}
.module-card:hover{border-color:var(--border2);transform:translateY(-3px);box-shadow:0 18px 44px rgba(0,0,0,.22)}
.module-icon{font-size:2.1rem;margin-bottom:.7rem}
.module-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:900;margin-bottom:.35rem}
.module-meta{color:var(--muted);font-size:.82rem;line-height:1.65;min-height:44px}

/* Doc pages */
.doc-page{
  background:rgba(13,19,39,.88);border:1px solid var(--border);border-radius:var(--r);
  padding:1.55rem;margin-bottom:1rem;box-shadow:0 12px 32px rgba(0,0,0,.14);
}
.doc-section-title{
  font-family:'Syne',sans-serif;font-size:.76rem;color:#93c5fd;
  letter-spacing:.14em;text-transform:uppercase;font-weight:900;margin-bottom:.95rem;
}
.doc-title{
  font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:900;
  color:#e5eefc;margin-bottom:1rem;line-height:1.3;
}
.doc-body{color:var(--text);line-height:2;font-size:.97rem}
.doc-body p{margin-bottom:.8rem}
.doc-body strong{color:#93c5fd;font-weight:700}

/* Chips */
.chip{display:inline-block;font-size:.72rem;font-weight:800;padding:5px 13px;border-radius:999px;margin:3px 2px}
.chip-blue  {background:rgba(59,130,246,.13);border:1px solid rgba(59,130,246,.30);color:#93c5fd}
.chip-green {background:rgba(16,185,129,.13);border:1px solid rgba(16,185,129,.30);color:#6ee7b7}
.chip-amber {background:rgba(245,158,11,.13); border:1px solid rgba(245,158,11,.30); color:#fcd34d}
.chip-rose  {background:rgba(244,63,94,.13);  border:1px solid rgba(244,63,94,.30);  color:#fda4af}
.chip-violet{background:rgba(139,92,246,.13); border:1px solid rgba(139,92,246,.30); color:#c4b5fd}
.chip-cyan  {background:rgba(6,182,212,.13);  border:1px solid rgba(6,182,212,.30);  color:#67e8f9}

/* Metrics */
.metric-row{display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0}
.metric-tile{
  flex:1;min-width:155px;background:rgba(17,26,49,.92);border:1px solid var(--border);
  border-radius:18px;padding:1.1rem;text-align:center;box-shadow:0 8px 22px rgba(0,0,0,.14);
}
.metric-val{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:900;color:#93c5fd}
.metric-lbl{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-top:.25rem}

/* Pipeline */
.pipeline{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:1rem 0}
.step{background:rgba(17,26,49,.9);border:1px solid var(--border);border-radius:10px;padding:7px 13px;color:var(--dim);font-weight:800;font-size:.72rem}
.step.done   {background:rgba(16,185,129,.1); border-color:rgba(16,185,129,.36);color:#6ee7b7}
.step.running{background:rgba(59,130,246,.1); border-color:rgba(59,130,246,.36);color:#93c5fd}
.step.error  {background:rgba(244,63,94,.1);  border-color:rgba(244,63,94,.36); color:#fda4af}

/* Key point card */
.kp-card{
  background:rgba(17,26,49,.85);border-left:3px solid #3b82f6;border-radius:0 12px 12px 0;
  padding:.85rem 1.1rem;margin:.5rem 0;
}
.kp-num{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--dim);margin-bottom:.25rem}
.kp-text{color:var(--text);font-size:.92rem;line-height:1.65}

/* Video card */
.video-card{
  background:rgba(13,19,39,.88);border:1px solid var(--border);border-radius:14px;
  padding:.85rem 1rem;margin:.45rem 0;display:flex;align-items:center;gap:.8rem;
  transition:border-color .15s;text-decoration:none;
}
.video-card:hover{border-color:var(--border2)}
.video-icon{font-size:1.6rem;flex-shrink:0}
.video-label{color:var(--muted);font-size:.82rem;margin-bottom:.1rem}
.video-title{color:#93c5fd;font-weight:700;font-size:.9rem;line-height:1.3}

/* Lang grid */
.lang-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin:1rem 0}
.lang-btn{
  background:rgba(17,26,49,.85);border:1px solid var(--border);border-radius:12px;
  padding:.7rem .9rem;cursor:pointer;transition:all .15s;text-align:center;
  font-family:'DM Sans',sans-serif;font-size:.88rem;color:var(--muted);
}
.lang-btn:hover{border-color:var(--border2);color:var(--text)}
.lang-btn.active{
  background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.50);
  color:#93c5fd;font-weight:700;
}

/* Translation display */
.translation-panel{
  background:rgba(13,19,39,.88);border:1px solid var(--border);border-radius:var(--r);
  padding:1.55rem;margin-bottom:1rem;
}
.translation-source{border-right:1px solid var(--border)}

/* HITL */
.hitl-box{
  background:linear-gradient(135deg,rgba(16,185,129,.10),rgba(6,182,212,.07));
  border:1px solid var(--border3);border-radius:var(--r);
  padding:1.45rem;margin:1.4rem 0;box-shadow:0 10px 28px rgba(0,0,0,.12);
}
.hitl-title{font-family:'Syne',sans-serif;font-weight:900;color:#6ee7b7;font-size:1.05rem;margin-bottom:.5rem}
.hitl-text{color:var(--muted);line-height:1.78;font-size:.9rem}

/* History */
.history-item{
  background:rgba(17,26,49,.92);border:1px solid var(--border);border-radius:16px;
  padding:1.1rem 1.2rem;margin-bottom:.8rem;transition:border-color .15s;
}
.history-item:hover{border-color:var(--border2)}
.small-note{color:var(--muted);font-size:.82rem;line-height:1.72}

/* Summary paragraph styling */
.summary-text{
  color:var(--text);line-height:2.1;font-size:.98rem;
  white-space:pre-wrap;word-break:break-word;
}
.summary-title{
  font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:900;
  color:#e5eefc;margin-bottom:1rem;padding-bottom:.75rem;
  border-bottom:1px solid var(--border);
}

/* ── Chat — style AI moderne ─────────────────────────────── */
.msg{display:flex;gap:1rem;align-items:flex-start;padding:1.2rem 0 1rem 0;border-bottom:1px solid rgba(255,255,255,.05);width:100%}
.msg.user{flex-direction:row-reverse;border-bottom:none;padding:.5rem 0}
.msg-avatar{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;margin-top:.1rem}
.msg-avatar.bot{background:linear-gradient(135deg,#2563eb,#7c3aed);box-shadow:0 2px 10px rgba(99,102,241,.4)}
.msg-avatar.user{background:linear-gradient(135deg,#0f766e,#0891b2);box-shadow:0 2px 8px rgba(8,145,178,.3)}
.msg-bubble{line-height:1.75;font-size:.93rem;max-width:100%;word-break:break-word;flex:1;min-width:0}
.msg-bubble.bot{color:var(--text);padding:0}
.msg-bubble.user{background:linear-gradient(135deg,rgba(37,99,235,.25),rgba(79,70,229,.2));border:1px solid rgba(99,102,241,.3);color:var(--text);border-radius:18px 4px 18px 18px;padding:.8rem 1.1rem;display:inline-block;max-width:88%}
.msg-time{font-size:.66rem;color:var(--dim);margin-top:.55rem;opacity:.65}
.msg.user .msg-time{text-align:right}
.chat-h2{font-size:1.01rem;font-weight:700;color:#93c5fd;margin:1rem 0 .4rem 0;padding-bottom:.3rem;border-bottom:1px solid rgba(99,102,241,.2);line-height:1.4}
.chat-h3{font-size:.94rem;font-weight:700;color:#a5b4fc;margin:.8rem 0 .3rem 0;line-height:1.4}
.chat-p{margin:.3rem 0;line-height:1.75;font-size:.93rem;color:var(--text)}
.chat-ul{margin:.4rem 0 .6rem 0;padding:0;list-style:none}
.chat-ul li{position:relative;padding:.25rem 0 .25rem 1.4rem;line-height:1.65;font-size:.93rem;color:var(--text)}
.chat-ul li::before{content:"▸";position:absolute;left:.15rem;color:#60a5fa;font-weight:700;font-size:.78rem;top:.32rem}
.chat-spacer{height:.5rem}
.chat-mode-badge{display:inline-block;font-size:.66rem;color:#818cf8;border:1px solid rgba(99,102,241,.28);border-radius:4px;padding:.08rem .38rem;margin-left:.5rem;vertical-align:middle;opacity:.8}
.msg-sources{
  margin-top:.6rem;padding:.6rem .8rem;
  background:rgba(59,130,246,.08);border-radius:10px;
  border-left:2px solid rgba(59,130,246,.40);
  font-size:.78rem;color:var(--muted);
}
.chat-mode-badge{
  font-family:'JetBrains Mono',monospace;font-size:.65rem;
  color:var(--dim);margin-left:.5rem;
}
.chat-welcome{
  background:linear-gradient(135deg,rgba(59,130,246,.10),rgba(139,92,246,.08));
  border:1px solid var(--border);border-radius:var(--r);padding:1.4rem;
  margin-bottom:1rem;text-align:center;
}
.chat-welcome-icon{font-size:2.5rem;margin-bottom:.6rem}
.chat-welcome-title{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:900;margin-bottom:.4rem}
.chat-welcome-text{color:var(--muted);font-size:.88rem;line-height:1.7}
.chat-suggestions{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem;justify-content:center}
.suggestion-chip{
  background:rgba(17,26,49,.85);border:1px solid var(--border);border-radius:999px;
  padding:6px 14px;font-size:.8rem;color:#93c5fd;cursor:pointer;
  transition:all .15s;text-decoration:none;
}
.suggestion-chip:hover{border-color:var(--border2);background:rgba(37,99,235,.15)}
.chat-stats{
  background:rgba(13,19,39,.88);border:1px solid var(--border);border-radius:14px;
  padding:.85rem 1.1rem;display:flex;gap:1.2rem;align-items:center;margin-bottom:1rem;
}
.chat-stat-item{text-align:center}
.chat-stat-val{font-family:'Syne',sans-serif;font-weight:900;font-size:1.1rem;color:#93c5fd}
.chat-stat-lbl{font-size:.67rem;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

DEFAULTS = {
    "page"            : "home",
    "analysis_result" : None,
    "current_file"    : None,
    "current_pdf_path": None,
    "step_states"     : {},
    "validated"       : False,
    "report_paths"    : {},
    "summary_remarks" : "",
    "validator_note"  : "",
    "trans_cache"     : {},      # cache {lang_code: translated_text}
    "trans_lang_ui"   : "fr",
    "chat_history"    : [],          # liste de {role, content, time, sources, mode}
    "chat_input_val"  : "",
    "chat_doc_meta"   : {},
   # langue sélectionnée sur la page traduction
    "do_explanation"  : True,
    "do_translation"  : True,
    "target_language" : "fr",
}

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def reset_analysis(keep_file: bool = False) -> None:
    current_file     = st.session_state.get("current_file")
    current_pdf_path = st.session_state.get("current_pdf_path")
    for _k, _v in DEFAULTS.items():
        st.session_state[_k] = _v
    if keep_file:
        st.session_state.current_file     = current_file
        st.session_state.current_pdf_path = current_pdf_path


# =============================================================================
# HISTORIQUE
# =============================================================================

def load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history_entry(result: dict, pdf_path: str) -> None:
    history = load_history()
    doc  = result.get("document", {})
    summ = result.get("summary",  {})
    trn  = result.get("translation", {})
    entry = {
        "timestamp"           : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name"           : doc.get("file_name") or Path(pdf_path).name,
        "pdf_path"            : pdf_path,
        "pages"               : doc.get("page_count", "—"),
        "words"               : doc.get("word_count", 0),
        "title"               : summ.get("title", ""),
        "summary_preview"     : (summ.get("summary", "") or "")[:260],
        "translation_language": trn.get("target_language", ""),
    }
    history.insert(0, entry)
    HISTORY_PATH.write_text(
        json.dumps(history[:30], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# HELPERS UI
# =============================================================================

STEPS = [
    ("extraction",    "① Extraction"),
    ("summarisation", "② Résumé"),
    ("explanation",   "③ Explication"),
    ("translation",   "④ Traduction"),
    ("hitl",          "⑤ HITL"),
    ("report",        "⑥ Rapport"),
]


def pipeline_html(states: dict) -> str:
    html = "<div class='pipeline'>"
    for key, label in STEPS:
        status = states.get(key, "pending")
        if   status == "done":    html += f"<div class='step done'>✓ {label}</div>"
        elif status == "running": html += f"<div class='step running'>◉ {label}</div>"
        elif status == "error":   html += f"<div class='step error'>✕ {label}</div>"
        else:                     html += f"<div class='step'>○ {label}</div>"
    return html + "</div>"


def result_or_stop(module_name: str) -> dict:
    result = st.session_state.analysis_result
    if not result:
        st.warning(
            f"Le module **{module_name}** n'a pas encore de résultat. "
            "Retournez au tableau de bord, uploadez un PDF puis lancez l'analyse."
        )
        if st.button("← Retour au tableau de bord"):
            go("home")
        st.stop()
    return result


def back_button() -> None:
    if st.button("← Retour au tableau de bord"):
        go("home")


def safe_download(path_value, label: str, mime: str) -> None:
    if not path_value:
        return
    p = Path(path_value)
    if p.exists():
        st.download_button(label, data=p.read_bytes(), file_name=p.name,
                           mime=mime, use_container_width=True)


def fmt_summary(text: str) -> str:
    """Formate le texte du résumé en paragraphes HTML lisibles."""
    import html as html_lib
    if not text:
        return "<em>Aucun résumé disponible.</em>"
    text = text.strip()
    # Sépare en paragraphes sur double saut de ligne ou après ". "
    paragraphs = [p.strip() for p in text.replace('\r','').split('\n\n') if p.strip()]
    if len(paragraphs) == 1:
        # Tente une découpe sur les points si pas de paragraphes
        import re
        parts = re.split(r'(?<=[.!?]) {2,}|\n', paragraphs[0])
        if len(parts) > 2:
            paragraphs = [p.strip() for p in parts if len(p.strip()) > 30]
    return "".join(f"<p>{html_lib.escape(p)}</p>" for p in paragraphs if p)


def yt_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)


def make_video_links(terms: list[str], category: str, title: str) -> list[dict]:
    """Génère des liens YouTube pertinents basés sur les termes et la catégorie."""
    queries = []

    # Sujet principal du document
    if title and len(title) > 10:
        queries.append({"label": "Sujet principal", "query": title, "icon": "🎯"})

    # Catégorie du document
    cat_queries = {
        "Course / Academic"       : "academic writing lecture tutorial",
        "Research Paper"          : "research methodology tutorial",
        "Report"                  : "how to write professional report",
        "Business / Invoice"      : "business document management",
        "Administrative"          : "administrative procedures tutorial",
        "Legal / Contract"        : "contract law basics explained",
        "Medical / Health"        : "medical terminology explained",
        "Technical / Engineering" : "engineering concepts tutorial",
    }
    if category in cat_queries:
        queries.append({"label": f"Catégorie : {category}", "query": cat_queries[category], "icon": "📚"})

    # Termes clés (les 4 premiers)
    for term in terms[:4]:
        if len(term) > 4:
            queries.append({"label": f"Terme : {term}", "query": f"{term} explained tutorial", "icon": "🔑"})

    return queries[:7]  # Max 7 liens


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚡ SmartDoc AI")
    st.caption("Analyse PDF multi-agents")
    st.markdown("---")

    if st.button("🏠 Tableau de bord",  use_container_width=True): go("home")
    if st.button("🕘 Voir l'historique", use_container_width=True): go("history")

    if st.session_state.analysis_result:
        st.markdown("### 📂 Pages")
        if st.button("📝 Résumé détaillé",  use_container_width=True): go("summary")
        if st.button("💡 Explication",       use_container_width=True): go("explanation")
        if st.button("🌍 Traduction",        use_container_width=True): go("translation")
        if st.button("📘 Rapport général",   use_container_width=True): go("general_report")
        if st.button("💬 Chat avec le document", use_container_width=True): go("chat")

    st.markdown("---")
    st.markdown("### ⚙️ Options d'analyse")

    do_explanation = st.toggle("💡 Activer l'explication", value=st.session_state.do_explanation)
    do_translation = st.toggle("🌍 Activer la traduction",  value=st.session_state.do_translation)

    target_language = st.selectbox(
        "Langue initiale de traduction",
        list(ALL_LANGUAGES.keys()),
        index=list(ALL_LANGUAGES.keys()).index(st.session_state.target_language),
        format_func=lambda x: f"{ALL_LANGUAGES[x]['flag']} {ALL_LANGUAGES[x]['name']}",
        disabled=not do_translation,
    )
    st.session_state.do_explanation = do_explanation
    st.session_state.do_translation = do_translation
    st.session_state.target_language = target_language

    st.caption("La classification PyTorch est utilisée en interne uniquement.")
    st.markdown("---")
    if st.button("🧹 Réinitialiser", use_container_width=True):
        reset_analysis()
        clear_logs()
        st.success("Session réinitialisée.")
        st.rerun()


# =============================================================================
# HERO
# =============================================================================

st.markdown(f"""
<div class="hero">
  <span class="hero-version">v{APP_VERSION}</span>
  <div class="hero-badge">UIR · Semestre 8 · Smart Document Analyst</div>
  <div class="hero-title">SmartDoc AI</div>
  <div class="hero-sub">
    Déposez un PDF. Le pipeline multi-agents extrait le contenu, génère un résumé clair et
    structuré, explique les idées complexes avec des ressources vidéo, traduit dans 12 langues
    à la volée, puis produit un rapport essentiel avec validation humaine (HITL).
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE — HISTORIQUE
# =============================================================================

if st.session_state.page == "history":
    back_button()
    st.markdown("<div class='section-title'>🕘 Historique des analyses</div>", unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.info("Aucune analyse enregistrée.")
        st.stop()
    for item in history:
        lang = item.get("translation_language", "")
        lang_str = f" · Traduction : {ALL_LANGUAGES.get(lang, {}).get('flag','')} {ALL_LANGUAGES.get(lang, {}).get('name', lang)}" if lang else ""
        title_str = f"<div style='color:var(--muted);font-size:.8rem;font-style:italic;margin-top:.3rem'>{item.get('title','')}</div>" if item.get("title") else ""
        st.markdown(f"""
<div class="history-item">
  <div style="font-family:'Syne',sans-serif;font-weight:900;font-size:1rem">📄 {item.get("file_name","Document")}</div>
  {title_str}
  <div class="small-note">{item.get("timestamp","—")} &nbsp;·&nbsp; {item.get("pages","—")} pages &nbsp;·&nbsp; {item.get("words",0):,} mots{lang_str}</div>
  <div class="small-note" style="margin-top:.5rem;font-style:italic">{item.get("summary_preview","")}</div>
</div>""", unsafe_allow_html=True)
    st.stop()


# =============================================================================
# PAGE — RÉSUMÉ DÉTAILLÉ
# =============================================================================

if st.session_state.page == "summary":
    back_button()
    result  = result_or_stop("Résumé")
    summary = result.get("summary", {})

    st.markdown("<div class='section-title'>📝 Résumé détaillé du document</div>", unsafe_allow_html=True)

    doc_title  = summary.get("title", "")
    sum_text   = (summary.get("summary") or "").strip()
    key_points = summary.get("key_points", [])
    mode_sum   = summary.get("mode", "")

    # ── En-tête du document ─────────────────────────────────────────────────
    doc     = result.get("document", {})
    cls     = result.get("classification", {})
    pages   = doc.get("page_count","—")
    words   = doc.get("word_count",0)
    cat     = cls.get("category","")
    mode_lbl= "🤖 Gemini" if mode_sum == "gemini" else "📊 Extractif"
    cat_chip= f"<span class='chip chip-cyan'>{cat}</span>" if cat else ""
    mode_chip= f"<span class='chip chip-violet'>{mode_lbl}</span>"

    st.markdown(f"""
<div class="doc-page">
  <div class="doc-section-title">Informations</div>
  <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
    <span class="chip chip-blue">📄 {pages} pages</span>
    <span class="chip chip-blue">{words:,} mots</span>
    {cat_chip}
    {mode_chip}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Titre + Résumé ──────────────────────────────────────────────────────
    title_html = f"<div class='summary-title'>{doc_title}</div>" if doc_title else ""

    if not sum_text:
        sum_text = "Aucun résumé n'a pu être généré pour ce document."

    st.markdown(f"""
<div class="doc-page">
  <div class="doc-section-title">Résumé détaillé</div>
  {title_html}
  <div class="doc-body">{fmt_summary(sum_text)}</div>
</div>""", unsafe_allow_html=True)

    # ── Points clés ─────────────────────────────────────────────────────────
    if key_points:
        st.markdown("<div class='section-title'>🎯 Points clés</div>", unsafe_allow_html=True)
        for i, kp in enumerate(key_points, 1):
            st.markdown(f"""
<div class="kp-card">
  <div class="kp-num">Point {i:02d}</div>
  <div class="kp-text">{kp}</div>
</div>""", unsafe_allow_html=True)

    # ── Remarques ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🗒️ Remarques personnelles</div>", unsafe_allow_html=True)
    st.session_state.summary_remarks = st.text_area(
        "Ajoutez vos remarques",
        value=st.session_state.summary_remarks,
        placeholder="Ex : Le document est clair, mais la partie méthodologie mérite vérification…",
        height=130,
    )

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📥 Exporter le résumé</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Générer résumé PDF", use_container_width=True):
            with st.spinner("Génération PDF…"):
                r = generate_summary_pdf_report(result, remarks=st.session_state.summary_remarks)
            if r.get("status") == "success":
                st.session_state.report_paths["summary_pdf"] = r["report_path"]
                st.success("PDF généré.")
            else:
                st.error(r.get("error"))
        safe_download(st.session_state.report_paths.get("summary_pdf"),
                      "⬇️ Télécharger résumé PDF", "application/pdf")
    with col2:
        if st.button("📝 Générer résumé Word", use_container_width=True):
            with st.spinner("Génération Word…"):
                r = generate_summary_docx_report(result, remarks=st.session_state.summary_remarks)
            if r.get("status") == "success":
                st.session_state.report_paths["summary_docx"] = r["report_path"]
                st.success("Word généré.")
            else:
                st.error(r.get("error"))
        safe_download(st.session_state.report_paths.get("summary_docx"),
                      "⬇️ Télécharger résumé Word",
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    st.stop()


# =============================================================================
# PAGE — EXPLICATION + VIDÉOS YOUTUBE
# =============================================================================

if st.session_state.page == "explanation":
    back_button()
    result      = result_or_stop("Explication")
    explanation = result.get("explanation", {})
    summary     = result.get("summary", {})
    cls         = result.get("classification", {})

    st.markdown("<div class='section-title'>💡 Explication du document</div>", unsafe_allow_html=True)

    if explanation.get("status") in ("skipped", "error") or not explanation.get("simple_explanation"):
        st.warning("L'explication n'était pas activée. Activez **💡 Explication** dans la sidebar et relancez.")
        st.stop()

    # ── Explication + niveau ─────────────────────────────────────────────────
    level      = explanation.get("reading_level", "")
    level_map  = {"Simple": "chip-green", "Intermédiaire": "chip-amber", "Avancé": "chip-rose"}
    level_chip = f"<span class='chip {level_map.get(level, 'chip-blue')}'>{level}</span>" if level else ""
    mode_ex    = explanation.get("mode", "")
    mode_chip  = f"<span class='chip chip-violet'>{'🤖 Gemini' if mode_ex == 'gemini' else '📊 Local'}</span>"

    st.markdown(f"""
<div class="doc-page">
  <div class="doc-section-title">Explication simplifiée &nbsp; {level_chip} {mode_chip}</div>
  <div class="doc-body">{fmt_summary(explanation.get("simple_explanation",""))}</div>
</div>""", unsafe_allow_html=True)

    # ── Termes importants ────────────────────────────────────────────────────
    terms = explanation.get("important_terms", [])
    if terms:
        st.markdown("<div class='section-title'>🏷️ Termes importants</div>", unsafe_allow_html=True)
        chips = "".join(
            f"<a href='https://fr.wikipedia.org/wiki/{urllib.parse.quote(t)}' target='_blank' "
            f"style='text-decoration:none'><span class='chip chip-blue' style='cursor:pointer'>🔗 {t}</span></a>"
            for t in terms
        )
        st.markdown(f"<div style='margin:.5rem 0'>{chips}</div>", unsafe_allow_html=True)

    # ── Vidéos YouTube ───────────────────────────────────────────────────────
    doc_title = summary.get("title", "")
    category  = cls.get("category", "")
    video_links = make_video_links(terms, category, doc_title)

    if video_links:
        st.markdown("<div class='section-title'>▶️ Ressources vidéo recommandées</div>", unsafe_allow_html=True)
        st.markdown("<div class='doc-page'><div class='doc-section-title'>Vidéos YouTube pour approfondir</div>", unsafe_allow_html=True)

        cols = st.columns(2)
        for i, vl in enumerate(video_links):
            url = yt_search_url(vl["query"])
            with cols[i % 2]:
                st.markdown(f"""
<a href="{url}" target="_blank" class="video-card" style="display:flex;align-items:center;gap:.8rem;text-decoration:none;background:rgba(17,26,49,.85);border:1px solid var(--border);border-radius:14px;padding:.85rem 1rem;margin:.4rem 0;transition:border-color .15s">
  <div style="font-size:1.6rem;flex-shrink:0">{vl["icon"]}</div>
  <div>
    <div style="color:var(--dim);font-size:.74rem;margin-bottom:.15rem">{vl["label"]}</div>
    <div style="color:#93c5fd;font-weight:700;font-size:.88rem;line-height:1.3">{vl["query"]}</div>
  </div>
  <div style="margin-left:auto;color:#ff0000;font-size:1.2rem">▶</div>
</a>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Questions d'étude ────────────────────────────────────────────────────
    questions = explanation.get("study_questions", [])
    if questions:
        st.markdown("<div class='section-title'>❓ Questions de compréhension</div>", unsafe_allow_html=True)
        st.markdown("<div class='doc-page'><div class='doc-section-title'>Questions pour préparer la soutenance</div>", unsafe_allow_html=True)
        for i, q in enumerate(questions, 1):
            st.markdown(f"""
<div class="kp-card" style="border-left-color:#8b5cf6">
  <div class="kp-num">Question {i:02d}</div>
  <div class="kp-text">{q}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# =============================================================================
# PAGE — TRADUCTION INTERACTIVE (12 langues)
# =============================================================================

if st.session_state.page == "translation":
    back_button()
    result  = result_or_stop("Traduction")
    summary = result.get("summary", {})
    source  = (summary.get("summary") or result.get("raw_text","")[:2_000]).strip()

    st.markdown("<div class='section-title'>🌍 Traduction — 12 langues disponibles</div>", unsafe_allow_html=True)

    if not source:
        st.warning("Aucun texte source disponible pour la traduction.")
        st.stop()

    # ── Sélecteur de langue interactif ──────────────────────────────────────
    st.markdown("<div class='doc-page'><div class='doc-section-title'>Choisissez la langue cible</div>", unsafe_allow_html=True)

    lang_cols = st.columns(6)
    lang_codes = list(ALL_LANGUAGES.keys())
    current_lang = st.session_state.trans_lang_ui

    for i, code in enumerate(lang_codes):
        info   = ALL_LANGUAGES[code]
        active = "active" if code == current_lang else ""
        with lang_cols[i % 6]:
            if st.button(
                f"{info['flag']} {info['name']}",
                use_container_width=True,
                key=f"lang_btn_{code}",
                type="primary" if code == current_lang else "secondary",
            ):
                st.session_state.trans_lang_ui = code
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    sel_lang = st.session_state.trans_lang_ui
    sel_info = ALL_LANGUAGES[sel_lang]

    # ── Traduction (cache par langue) ────────────────────────────────────────
    trans_cache = st.session_state.trans_cache

    # Pré-charge depuis l'analyse initiale si langue identique
    initial_tr = result.get("translation", {})
    if (initial_tr.get("status") == "success"
            and initial_tr.get("target_language") == sel_lang
            and sel_lang not in trans_cache):
        trans_cache[sel_lang] = initial_tr.get("translated_text", "")

    needs_translation = sel_lang not in trans_cache

    if needs_translation:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            if st.button(f"Traduire en {sel_info['flag']} {sel_info['name']}",
                         type="primary", use_container_width=True):
                with st.spinner(f"Traduction en {sel_info['name']} en cours…"):
                    tr = translate_text(source, sel_lang)
                if tr.get("status") == "success":
                    trans_cache[sel_lang] = tr["translated_text"]
                    st.session_state.trans_cache = trans_cache
                    st.success("Traduction disponible.")
                    st.rerun()
                else:
                    st.error(f"Erreur : {tr.get('error')}")
        with col_info:
            st.info(f"Cliquez sur le bouton pour traduire le résumé en **{sel_info['name']}**.")
    else:
        # Affichage côte à côte
        direction = sel_info.get("dir", "ltr")
        translated = trans_cache.get(sel_lang, "")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
<div class="doc-page">
  <div class="doc-section-title">Texte source</div>
  <div class="doc-body">{fmt_summary(source)}</div>
</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
<div class="doc-page">
  <div class="doc-section-title">{sel_info['flag']} Traduction — {sel_info['name']}</div>
  <div class="doc-body" dir="{direction}">{fmt_summary(translated)}</div>
</div>""", unsafe_allow_html=True)

        # Bouton pour re-traduire
        if st.button(f"🔄 Re-traduire en {sel_info['flag']} {sel_info['name']}", use_container_width=False):
            with st.spinner("Re-traduction en cours…"):
                tr = translate_text(source, sel_lang)
            if tr.get("status") == "success":
                trans_cache[sel_lang] = tr["translated_text"]
                st.session_state.trans_cache = trans_cache
                st.rerun()
            else:
                st.error(f"Erreur : {tr.get('error')}")

        # Toutes les traductions effectuées
        done_langs = list(trans_cache.keys())
        if len(done_langs) > 1:
            st.markdown("<div class='section-title'>✅ Traductions disponibles</div>", unsafe_allow_html=True)
            chips = "".join(
                f"<span class='chip {'chip-green' if c == sel_lang else 'chip-blue'}'>"
                f"{ALL_LANGUAGES[c]['flag']} {ALL_LANGUAGES[c]['name']}</span>"
                for c in done_langs
            )
            st.markdown(chips, unsafe_allow_html=True)

        # ── Export ───────────────────────────────────────────────────────────
        # Crée un résultat temporaire avec la traduction sélectionnée pour l'export
        export_result = dict(result)
        export_result["translation"] = {
            "status"         : "success",
            "translated_text": translated,
            "source_text"    : source,
            "target_language": sel_lang,
            "mode"           : "cached",
        }

        st.markdown("<div class='section-title'>📥 Exporter la traduction</div>", unsafe_allow_html=True)
        colp, cold = st.columns(2)
        with colp:
            if st.button("📄 Générer traduction PDF", use_container_width=True):
                with st.spinner("Génération PDF…"):
                    r = generate_translation_pdf_report(export_result)
                if r.get("status") == "success":
                    st.session_state.report_paths["translation_pdf"] = r["report_path"]
                    st.success("Traduction PDF générée.")
                else:
                    st.error(r.get("error"))
            safe_download(st.session_state.report_paths.get("translation_pdf"),
                          "⬇️ Télécharger traduction PDF", "application/pdf")
        with cold:
            if st.button("📝 Générer traduction Word", use_container_width=True):
                with st.spinner("Génération Word…"):
                    r = generate_translation_docx_report(export_result)
                if r.get("status") == "success":
                    st.session_state.report_paths["translation_docx"] = r["report_path"]
                    st.success("Traduction Word générée.")
                else:
                    st.error(r.get("error"))
            safe_download(st.session_state.report_paths.get("translation_docx"),
                          "⬇️ Télécharger traduction Word",
                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    st.stop()


# =============================================================================
# PAGE — RAPPORT GÉNÉRAL (HITL)
# =============================================================================

if st.session_state.page == "general_report":
    back_button()
    result = result_or_stop("Rapport général")

    st.markdown("<div class='section-title'>📘 Rapport général essentiel</div>", unsafe_allow_html=True)

    st.markdown("""
<div class="doc-page">
  <div class="doc-section-title">Contenu du rapport</div>
  <div class="doc-body">
    <p>Le rapport général essentiel rassemble : le résumé structuré, vos remarques personnelles,
    l'explication simplifiée (si activée), la traduction (si disponible), et votre note de validation humaine.</p>
    <p>La classification interne PyTorch n'est pas incluse dans le rapport final.</p>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="hitl-box">
  <div class="hitl-title">🤝 Validation humaine obligatoire (HITL)</div>
  <div class="hitl-text">
    Vérifiez les résultats sur les pages Résumé, Explication et Traduction
    avant de générer le rapport. Votre note sera incluse dans le document final.
  </div>
</div>""", unsafe_allow_html=True)

    st.session_state.validator_note = st.text_area(
        "Note de validation",
        value=st.session_state.validator_note,
        placeholder="Ex : Le résumé est correct, la traduction validée, le document est conforme aux attentes.",
        height=120,
    )

    # Prépare le résultat avec la meilleure traduction disponible
    best_result = dict(result)
    cache = st.session_state.trans_cache
    if cache:
        best_lang = st.session_state.trans_lang_ui if st.session_state.trans_lang_ui in cache else list(cache.keys())[0]
        best_result["translation"] = {
            "status": "success",
            "translated_text": cache[best_lang],
            "target_language": best_lang,
            "mode": "cached",
        }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Générer rapport général PDF", type="primary", use_container_width=True):
            with st.spinner("Génération PDF…"):
                r = generate_general_pdf_report(
                    best_result,
                    remarks=st.session_state.summary_remarks,
                    validator_note=st.session_state.validator_note,
                )
            if r.get("status") == "success":
                st.session_state.validated = True
                st.session_state.report_paths["general_pdf"] = r["report_path"]
                st.session_state.step_states.update({"hitl": "done", "report": "done"})
                st.success("✅ Rapport général PDF généré.")
            else:
                st.error(r.get("error"))
        safe_download(st.session_state.report_paths.get("general_pdf"),
                      "⬇️ Télécharger rapport PDF", "application/pdf")
    with col2:
        if st.button("📝 Générer rapport général Word", type="primary", use_container_width=True):
            with st.spinner("Génération Word…"):
                r = generate_general_docx_report(
                    best_result,
                    remarks=st.session_state.summary_remarks,
                    validator_note=st.session_state.validator_note,
                )
            if r.get("status") == "success":
                st.session_state.validated = True
                st.session_state.report_paths["general_docx"] = r["report_path"]
                st.session_state.step_states.update({"hitl": "done", "report": "done"})
                st.success("✅ Rapport général Word généré.")
            else:
                st.error(r.get("error"))
        safe_download(st.session_state.report_paths.get("general_docx"),
                      "⬇️ Télécharger rapport Word",
                      "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    st.stop()



# =============================================================================
# PAGE — CHAT DOCUMENTAIRE
# =============================================================================

if st.session_state.page == "chat":
    back_button()
    result = result_or_stop("Chat")

    doc          = result.get("document",       {})
    summary_data = result.get("summary",        {})
    cls          = result.get("classification", {})

    raw_text  = result.get("raw_text", "")
    doc_title = summary_data.get("title", doc.get("file_name","Document"))
    category  = cls.get("category", "")
    sum_text  = summary_data.get("summary", "")

    meta = {
        "title"     : doc_title,
        "category"  : category,
        "summary"   : sum_text[:1_200],
        "page_count": doc.get("page_count","?"),
        "word_count": doc.get("word_count","?"),
        "file_name" : doc.get("file_name",""),
    }

    # ── En-tête ──────────────────────────────────────────────────────────────
    from src.utils.config import GEMINI_API_KEY as _GEMINI_KEY, GROQ_API_KEY as _GROQ_KEY
    if _GROQ_KEY:
        mode_chip = "<span class=\'chip chip-green\'>✦ Groq — Llama 3.3</span>"
    elif _GEMINI_KEY:
        mode_chip = "<span class=\'chip chip-green\'>🤖 Gemini activé</span>"
    else:
        mode_chip = "<span class=\'chip chip-amber\'>📊 Mode local</span>"

    st.markdown(f"""
<div class="section-title">💬 Chat avec le document</div>
<div class="chat-stats">
  <div style="flex:1">
    <div class="chat-stat-val">📄 {doc.get("page_count","?")} pages</div>
    <div class="chat-stat-lbl">Document</div>
  </div>
  <div style="flex:1">
    <div class="chat-stat-val">{doc.get("word_count",0):,}</div>
    <div class="chat-stat-lbl">Mots</div>
  </div>
  <div style="flex:1">
    <div class="chat-stat-val">{len(st.session_state.chat_history)}</div>
    <div class="chat-stat-lbl">Échanges</div>
  </div>
  <div style="flex:2;text-align:left;padding-left:1rem">
    <div style="font-family:\'Syne\',sans-serif;font-size:.92rem;font-weight:900;margin-bottom:.3rem">{doc_title}</div>
    <div>{mode_chip}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Colonne chat + sidebar doc ────────────────────────────────────────────
    col_chat, col_doc = st.columns([2, 1])

    with col_doc:
        st.markdown("<div class=\'section-title\'>📎 Document</div>", unsafe_allow_html=True)

        # Infos document
        st.markdown(f"""
<div class="doc-page" style="padding:1rem 1.2rem;margin-bottom:.75rem">
  <div class="doc-section-title">Fichier analysé</div>
  <div class="card-text">
    <strong>{doc.get("file_name","")}</strong><br>
    {doc.get("page_count","?")} pages · {doc.get("word_count",0):,} mots
    {"<br><span class=\'chip chip-cyan\' style=\'margin-top:.4rem;display:inline-block\'>" + category + "</span>" if category else ""}
  </div>
</div>""", unsafe_allow_html=True)

        # Résumé rapide
        if sum_text:
            with st.expander("📝 Résumé du document", expanded=False):
                st.markdown(f"<div class=\'small-note\' style=\'line-height:1.7\'>{sum_text[:600]}...</div>",
                            unsafe_allow_html=True)

        # Questions suggérées
        st.markdown("<div class=\'section-title\'>💡 Questions suggérées</div>", unsafe_allow_html=True)

        suggestions = [
            "Quel est l\'objectif principal de ce document ?",
            "Quels sont les points essentiels ?",
            "Quelles sont les conclusions ?",
            "Quels outils ou technologies sont mentionnés ?",
            "Quelles sont les exigences minimales ?",
            "Quelle est la structure du document ?",
        ]

        for sug in suggestions:
            if st.button(f"💭 {sug}", use_container_width=True, key=f"sug_{sug[:20]}"):
                st.session_state.chat_history.append({
                    "role"   : "user",
                    "content": sug,
                    "time"   : datetime.now().strftime("%H:%M"),
                })
                with st.spinner("Analyse en cours…"):
                    resp = chat_agent.run(
                        question=sug,
                        document_text=raw_text,
                        meta=meta,
                        history=[m for m in st.session_state.chat_history[:-1]
                                 if m["role"] in ("user","assistant")],
                    )
                st.session_state.chat_history.append({
                    "role"   : "assistant",
                    "content": resp.get("answer","Désolé, je n'ai pas pu répondre."),
                    "time"   : datetime.now().strftime("%H:%M"),
                    "sources": resp.get("sources", []),
                    "mode"   : resp.get("mode","?"),
                })
                st.rerun()

        # Bouton reset
        st.markdown("---")
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with col_chat:
        st.markdown("<div class=\'section-title\'>💬 Conversation</div>", unsafe_allow_html=True)

        # Message de bienvenue si pas d'historique
        if not st.session_state.chat_history:
            st.markdown(f"""
<div class="chat-welcome">
  <div class="chat-welcome-icon">🤖</div>
  <div class="chat-welcome-title">Posez vos questions sur le document</div>
  <div class="chat-welcome-text">
    Je suis votre assistant documentaire. Posez-moi n'importe quelle question
    sur <strong>{doc_title}</strong> et je vous répondrai à partir de son contenu.
    {"<br><em>✨ Gemini activé — réponses précises et contextuelles.</em>" if _GEMINI_KEY else "<br><em>⚠️ Mode local — ajoutez GEMINI_API_KEY pour de meilleures réponses.</em>"}
  </div>
</div>""", unsafe_allow_html=True)
        else:
            # Zone des messages
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    role    = msg["role"]
                    content = msg["content"]
                    time    = msg.get("time","")
                    sources = msg.get("sources", [])
                    mode    = msg.get("mode","")

                    if role == "user":
                        import html as _html_u
                        safe_content = _html_u.escape(content)
                        st.markdown(
                            f'<div class="msg user">'
                            f'<div style="display:flex;flex-direction:column;align-items:flex-end">'
                            f'<div class="msg-bubble user">{safe_content}</div>'
                            f'<div class="msg-time">{time}</div>'
                            f'</div>'
                            f'<div class="msg-avatar user">👤</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        import html as _html, re as _re

                        def _md_to_html(text: str) -> str:
                            """Convertit markdown simple en HTML structuré."""
                            lines   = text.split("\n")
                            out     = []
                            in_list = False
                            for line in lines:
                                raw = line.rstrip()
                                # Titre ## ou ###
                                if raw.startswith("### "):
                                    if in_list: out.append("</ul>"); in_list = False
                                    h = _html.escape(raw[4:].strip())
                                    h = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", h)
                                    out.append(f'<div class="chat-h3">{h}</div>')
                                elif raw.startswith("## "):
                                    if in_list: out.append("</ul>"); in_list = False
                                    h = _html.escape(raw[3:].strip())
                                    h = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", h)
                                    out.append(f'<div class="chat-h2">{h}</div>')
                                # Puce - ou • ou *
                                elif _re.match(r"^[-•\*]\s+", raw):
                                    if not in_list: out.append('<ul class="chat-ul">'); in_list = True
                                    item = _html.escape(_re.sub(r"^[-•\*]\s+", "", raw).strip())
                                    item = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
                                    out.append(f'<li>{item}</li>')
                                # Ligne vide
                                elif raw == "":
                                    if in_list: out.append("</ul>"); in_list = False
                                    out.append('<div class="chat-spacer"></div>')
                                # Paragraphe normal
                                else:
                                    if in_list: out.append("</ul>"); in_list = False
                                    p = _html.escape(raw)
                                    p = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
                                    p = _re.sub(r"\*(.+?)\*",     r"<em>\1</em>",          p)
                                    out.append(f'<p class="chat-p">{p}</p>')
                            if in_list:
                                out.append("</ul>")
                            return "".join(out)

                        content_html = _md_to_html(content)
                        mode_badge   = (
                            f"<span class=\'chat-mode-badge\'>✦ {mode}</span>"
                            if mode and mode not in ("error",) else ""
                        )
                        sources_html = ""
                        if sources:
                            src_items = "".join(
                                f"<div style=\'margin:.25rem 0\'>→ {_html.escape(s[:160])}…</div>"
                                for s in sources[:3]
                            )
                            sources_html = f"<div class=\'msg-sources\'>📌 Extraits utilisés :{src_items}</div>"

                        st.markdown(
                            f'<div class="msg bot">'
                            f'<div class="msg-avatar bot">✦</div>'
                            f'<div class="msg-bubble bot">'
                            f'{content_html}'
                            f'{sources_html}'
                            f'<div style="display:flex;align-items:center;gap:.5rem;margin-top:.55rem">'
                            f'{mode_badge}'
                            f'<span class="msg-time">{time}</span>'
                            f'</div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ── Zone de saisie ────────────────────────────────────────────────────
        st.markdown("<div style=\'height:.8rem\'></div>", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([5, 1])
            with col_input:
                user_input = st.text_input(
                    "Question",
                    placeholder="Ex : Quels sont les objectifs du projet ? Qui est l\'auteur ?",
                    label_visibility="collapsed",
                    key="chat_input_field",
                )
            with col_send:
                submitted = st.form_submit_button("Envoyer ➤", use_container_width=True)

        if submitted and user_input.strip():
            q = user_input.strip()

            # Ajoute la question à l'historique
            st.session_state.chat_history.append({
                "role"   : "user",
                "content": q,
                "time"   : datetime.now().strftime("%H:%M"),
            })

            # Récupère la réponse
            with st.spinner("⟳ Recherche dans le document…"):
                history_for_agent = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                    if m["role"] in ("user","assistant")
                ][-12:]  # 12 derniers messages max

                resp = chat_agent.run(
                    question=q,
                    document_text=raw_text,
                    meta=meta,
                    history=history_for_agent,
                )

            st.session_state.chat_history.append({
                "role"   : "assistant",
                "content": resp.get("answer","Désolé, je n'ai pas pu répondre."),
                "time"   : datetime.now().strftime("%H:%M"),
                "sources": resp.get("sources", []),
                "mode"   : resp.get("mode","?"),
            })
            st.rerun()

    st.stop()


# =============================================================================
# PAGE PRINCIPALE — TABLEAU DE BORD (HOME)
# =============================================================================

st.markdown("<div class='section-title'>📄 Analyse d'un document PDF</div>", unsafe_allow_html=True)

st.markdown("""
<div class="upload-box">
  <div class="card-title">Déposez votre fichier PDF ici</div>
  <div class="card-text">
    Après l'upload, configurez les options dans la sidebar puis cliquez sur
    <strong>⚡ Analyser le document</strong>. Les résultats restent disponibles
    lors de la navigation entre les pages.
  </div>
</div>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Uploader un document PDF", type=["pdf"],
                                  label_visibility="collapsed")

if uploaded_file is not None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOADS_DIR / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getbuffer())
    if st.session_state.current_file != uploaded_file.name:
        reset_analysis()
        st.session_state.current_file     = uploaded_file.name
        st.session_state.current_pdf_path = str(pdf_path)
elif st.session_state.current_pdf_path:
    pdf_path = Path(st.session_state.current_pdf_path)
else:
    pdf_path = None


# ── Sans fichier — écran d'accueil ───────────────────────────────────────────

if pdf_path is None:
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("🧠","Classification DL","Modèle PyTorch TF-IDF + réseau FC — catégorise le document en interne."),
        ("📝","Résumé structuré","Résumé clair, titré, en paragraphes lisibles + 6 points clés."),
        ("🌍","Traduction ×12","Traduit à la volée dans 12 langues sans relancer l'analyse."),
        ("📘","Rapport HITL","Rapport final PDF/Word validé par l'opérateur humain."),
    ]
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],cards):
        with col:
            st.markdown(f"""
<div class="card">
  <div style="font-size:2.1rem;margin-bottom:.65rem">{icon}</div>
  <div class="card-title">{title}</div>
  <div class="card-text">{desc}</div>
</div>""", unsafe_allow_html=True)
    if load_history():
        _, c_mid, _ = st.columns([1,2,1])
        with c_mid:
            if st.button("🕘 Voir l'historique des analyses", use_container_width=True): go("history")
    st.info("Commencez par déposer un fichier PDF dans la zone ci-dessus.")
    st.stop()


# ── Fichier + options ────────────────────────────────────────────────────────

col_file, col_action = st.columns([1,2])

with col_file:
    try:
        size_kb = f"{Path(pdf_path).stat().st_size/1024:.1f} KB"
    except Exception:
        size_kb = "—"
    st.markdown(f"""
<div class="card">
  <div class="card-title">📎 Document sélectionné</div>
  <div class="card-text">
    <strong>Nom :</strong> {Path(pdf_path).name}<br>
    <strong>Taille :</strong> {size_kb}<br>
    <strong>Explication :</strong> {"✅ Activée" if do_explanation else "⛔ Désactivée"}<br>
    <strong>Traduction :</strong> {ALL_LANGUAGES[target_language]['flag']+' '+ALL_LANGUAGES[target_language]['name'] if do_translation else "⛔ Désactivée"}
  </div>
</div>""", unsafe_allow_html=True)

with col_action:
    st.markdown("""
<div class="card">
  <div class="card-title">🚀 Lancer l'analyse</div>
  <div class="card-text">
    Pipeline : extraction → classification PyTorch → résumé structuré →
    explication avec vidéos → traduction → rapport HITL.
  </div>
</div>""", unsafe_allow_html=True)
    run_button = st.button("⚡ Analyser le document", type="primary", use_container_width=True)


# ── Pipeline display ─────────────────────────────────────────────────────────

pipeline_placeholder = st.empty()
pipeline_placeholder.markdown(pipeline_html(st.session_state.step_states), unsafe_allow_html=True)


# ── Exécution ────────────────────────────────────────────────────────────────

if run_button:
    reset_analysis(keep_file=True)
    clear_logs()

    progress_bar = st.progress(0)
    status_box   = st.empty()
    steps_state  = {}

    STEP_MAP = {
        "extraction"      : "extraction",
        "classification"  : "extraction",
        "summarisation"   : "summarisation",
        "summarization"   : "summarisation",
        "explanation"     : "explanation",
        "translation"     : "translation",
        "human_validation": "hitl",
        "report"          : "report",
    }
    STEP_LABELS = {
        "extraction"      : "Extraction du contenu",
        "classification"  : "Classification interne",
        "summarisation"   : "Résumé détaillé",
        "summarization"   : "Résumé détaillé",
        "explanation"     : "Explication + vidéos",
        "translation"     : "Traduction initiale",
        "human_validation": "Validation HITL",
        "report"          : "Rapport final",
    }

    def progress_callback(step, status, pct):
        key = STEP_MAP.get(step, step)
        steps_state[key] = status
        st.session_state.step_states = dict(steps_state)
        pipeline_placeholder.markdown(pipeline_html(st.session_state.step_states), unsafe_allow_html=True)
        progress_bar.progress(min(max(float(pct), 0.0), 1.0))
        label = STEP_LABELS.get(step, step.replace("_"," ").title())
        if   status == "running": status_box.info(f"⏳ {label} en cours…")
        elif status == "done":    status_box.success(f"✅ {label} terminé.")
        elif status == "error":   status_box.error(f"❌ Erreur : {label}")

    try:
        orchestrator = SmartDocOrchestrator(progress_cb=progress_callback)
        result = orchestrator.analyze(
            str(pdf_path),
            do_summary     = True,
            do_explanation = do_explanation,
            do_translation = do_translation,
            target_language= target_language,
        )

        st.session_state.analysis_result = result

        if result.get("status") == "success":
            # Pré-charge le cache de traduction avec la traduction initiale
            initial_tr = result.get("translation", {})
            if initial_tr.get("status") == "success" and initial_tr.get("translated_text"):
                st.session_state.trans_cache = {
                    initial_tr.get("target_language", target_language): initial_tr.get("translated_text","")
                }
                st.session_state.trans_lang_ui = initial_tr.get("target_language", target_language)

            st.session_state.step_states["hitl"] = "pending"
            progress_bar.progress(0.97)
            status_box.success("✅ Analyse terminée. Naviguez entre les modules via la sidebar.")
            save_history_entry(result, str(pdf_path))
        else:
            st.session_state.step_states[result.get("step","extraction")] = "error"
            status_box.error(f"❌ Erreur à l'étape **{result.get('step','?')}** : {result.get('error','Erreur inconnue')}")

        pipeline_placeholder.markdown(pipeline_html(st.session_state.step_states), unsafe_allow_html=True)

    except Exception as exc:
        st.session_state.analysis_result = {"status":"error","step":"application","error":str(exc)}
        st.error(f"❌ Erreur critique : {exc}")


# ── Affichage résultats ──────────────────────────────────────────────────────

result = st.session_state.analysis_result

if not result:
    st.info("Cliquez sur **⚡ Analyser le document** pour lancer le pipeline.")
    st.stop()

if result.get("status") != "success":
    st.error(f"❌ Erreur à l'étape `{result.get('step','?')}` : {result.get('error','Erreur inconnue')}")
    st.stop()

document    = result.get("document",    {})
summary     = result.get("summary",     {})
explanation = result.get("explanation", {})
translation = result.get("translation", {})

st.success("✅ Analyse disponible — naviguez via la sidebar pour accéder aux modules.")

# Métriques
cache_langs = list(st.session_state.trans_cache.keys())
tr_lbl = " / ".join(ALL_LANGUAGES[c]["flag"] for c in cache_langs) if cache_langs else "—"

st.markdown(f"""
<div class="metric-row">
  <div class="metric-tile"><div class="metric-val">{document.get("page_count","—")}</div><div class="metric-lbl">Pages</div></div>
  <div class="metric-tile"><div class="metric-val">{document.get("word_count",0):,}</div><div class="metric-lbl">Mots</div></div>
  <div class="metric-tile"><div class="metric-val">{len(summary.get("summary","").split())}</div><div class="metric-lbl">Mots résumé</div></div>
  <div class="metric-tile"><div class="metric-val">{tr_lbl if tr_lbl != "—" else "—"}</div><div class="metric-lbl">Traductions</div></div>
</div>""", unsafe_allow_html=True)

# Aperçu du titre
doc_title = summary.get("title","")
if doc_title:
    st.markdown(f"""
<div class="doc-page" style="padding:1rem 1.4rem;margin-bottom:1rem">
  <div class="doc-section-title">Titre détecté</div>
  <div class="doc-title">{doc_title}</div>
</div>""", unsafe_allow_html=True)

# Modules
st.markdown("<div class='section-title'>📂 Modules disponibles</div>", unsafe_allow_html=True)
m1,m2,m3,m4 = st.columns(4)

with m1:
    st.markdown("""
<div class="module-card">
  <div class="module-icon">📝</div>
  <div class="module-title">Résumé détaillé</div>
  <div class="module-meta">Résumé structuré en paragraphes + 6 points clés + export.</div>
  <span class="chip chip-green">Disponible</span>
</div>""", unsafe_allow_html=True)
    if st.button("Ouvrir Résumé", use_container_width=True): go("summary")

with m2:
    exp_ok = explanation.get("status") == "success"
    st.markdown(f"""
<div class="module-card">
  <div class="module-icon">💡</div>
  <div class="module-title">Explication + Vidéos</div>
  <div class="module-meta">Explication, termes, liens YouTube, questions d'étude.</div>
  <span class="chip {'chip-green' if exp_ok else 'chip-amber'}">{'Disponible' if exp_ok else 'Non activée'}</span>
</div>""", unsafe_allow_html=True)
    if st.button("Ouvrir Explication", use_container_width=True): go("explanation")

with m3:
    n_langs = len(st.session_state.trans_cache)
    tr_chip = f"chip-green" if n_langs > 0 else "chip-blue"
    tr_lbl2 = f"{n_langs} langue{'s' if n_langs > 1 else ''}" if n_langs > 0 else "12 langues"
    st.markdown(f"""
<div class="module-card">
  <div class="module-icon">🌍</div>
  <div class="module-title">Traduction ×12</div>
  <div class="module-meta">Changez de langue à la volée sans relancer l'analyse.</div>
  <span class="chip {tr_chip}">{tr_lbl2}</span>
</div>""", unsafe_allow_html=True)
    if st.button("Ouvrir Traduction", use_container_width=True): go("translation")

with m4:
    validated = st.session_state.validated
    st.markdown(f"""
<div class="module-card">
  <div class="module-icon">📘</div>
  <div class="module-title">Rapport général</div>
  <div class="module-meta">Rapport final HITL avec note de validation.</div>
  <span class="chip {'chip-green' if validated else 'chip-blue'}">{'Validé ✓' if validated else 'PDF / Word'}</span>
</div>""", unsafe_allow_html=True)
    if st.button("Ouvrir Rapport", use_container_width=True): go("general_report")

# Ligne 2 : Chat
st.markdown("<div class='section-title'>💬 Assistant documentaire</div>", unsafe_allow_html=True)
chat_col, _ = st.columns([2, 2])
with chat_col:
    n_chat = len(st.session_state.chat_history)
    chat_badge = f"{n_chat} échange{'s' if n_chat > 1 else ''}" if n_chat else "Prêt"
    st.markdown(f"""
<div class="module-card" style="min-height:120px">
  <div class="module-icon">💬</div>
  <div class="module-title">Chat avec le document</div>
  <div class="module-meta">Posez des questions en langage naturel sur le contenu du PDF.</div>
  <span class="chip {'chip-green' if n_chat else 'chip-blue'}">{chat_badge}</span>
</div>""", unsafe_allow_html=True)
    if st.button("Ouvrir le Chat", use_container_width=True, type="primary"): go("chat")
