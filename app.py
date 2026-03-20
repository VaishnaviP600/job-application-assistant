import streamlit as st
from utils.pdf_processor import extract_text_from_pdf
from utils.rag_pipeline import RAGPipeline
from utils.prompts import SKILL_GAP_PROMPT, RESUME_REWRITE_PROMPT, COVER_LETTER_PROMPT

st.set_page_config(page_title="JobFit AI", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg: #0a0a0f; --surface: #13131a; --border: #1e1e2e;
    --accent: #7c3aed; --accent2: #06b6d4; --text: #e2e2f0;
    --muted: #6b6b8a; --success: #10b981;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg); color: var(--text); }
.main { background-color: var(--bg); }
.block-container { padding: 2rem 3rem; max-width: 1200px; }
h1,h2,h3 { font-family: 'Space Mono', monospace; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #06b6d4, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; }
.hero-sub { font-size: 1rem; color: var(--muted); margin-bottom: 2rem; font-weight: 300; }
.card-header { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: var(--accent2);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
.result-box { background: #0d0d14; border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 8px; padding: 1.2rem 1.5rem; font-size: 0.93rem; line-height: 1.7;
    white-space: pre-wrap; word-wrap: break-word; margin-top: 0.5rem; }
.result-box.cyan { border-left-color: var(--accent2); }
.result-box.green { border-left-color: var(--success); }
.status-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
    border-radius: 20px; font-size: 0.78rem; font-weight: 500; }
.status-pill.success { background: rgba(16,185,129,0.12); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.25); }
.stButton > button { background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important;
    padding: 0.6rem 1.5rem !important; letter-spacing: 0.05em !important; }
.stTextArea textarea { background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
hr { border-color: var(--border) !important; }
.step-indicator { display: flex; gap: 0.5rem; margin-bottom: 2rem; flex-wrap: wrap; }
.step { font-family: 'Space Mono', monospace; font-size: 0.72rem; padding: 5px 14px;
    border-radius: 6px; border: 1px solid var(--border); color: var(--muted); }
.step.active { border-color: var(--accent); color: #a78bfa; background: rgba(124,58,237,0.1); }
.step.done { border-color: var(--success); color: #6ee7b7; background: rgba(16,185,129,0.08); }
</style>
""", unsafe_allow_html=True)

# Session state
for key in ["resume_text", "jd_text", "results", "step"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key in ["resume_text", "jd_text"] else ({} if key == "results" else 0)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="card-header">⚙️ CONFIGURATION</div>', unsafe_allow_html=True)

    llm_provider = st.selectbox("LLM Provider", ["Groq (Free — Llama-3)", "OpenAI (GPT-4o)", "OpenAI (GPT-3.5-turbo)"], index=0)

    if "Groq" in llm_provider:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...",
            help="Free at console.groq.com/keys")
        provider = "groq"
        model_name = "llama3-8b-8192"
    elif "GPT-4o" in llm_provider:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        provider = "openai"
        model_name = "gpt-4o"
    else:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        provider = "openai"
        model_name = "gpt-3.5-turbo"

    vector_store = st.selectbox("Vector Store", ["FAISS", "ChromaDB"])
    st.markdown("---")
    st.markdown('<div class="card-header">📊 ANALYSIS OPTIONS</div>', unsafe_allow_html=True)
    run_skill_gap = st.checkbox("Skill Gap Analysis", value=True)
    run_resume_rewrite = st.checkbox("Resume Bullet Rewrites", value=True)
    run_cover_letter = st.checkbox("Cover Letter Generation", value=True)
    st.markdown("---")
    if "Groq" in llm_provider:
        st.markdown("""
        <div style='font-size:0.75rem; color:#6b6b8a; line-height:1.8;'>
        🆓 Groq is <b style='color:#6ee7b7'>completely free</b><br>
        Get key at <b>console.groq.com</b><br>
        Uses Llama-3 8B — fast & accurate
        </div>""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">JobFit AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">RAG-Powered Job Application Assistant — identify skill gaps, rewrite resume bullets, generate a tailored cover letter.</div>', unsafe_allow_html=True)

steps_html = ""
for i, label in enumerate(["01 · UPLOAD", "02 · ANALYZE", "03 · RESULTS"]):
    cls = "done" if st.session_state.step > i else ("active" if st.session_state.step == i else "")
    steps_html += f'<div class="step {cls}">{label}</div>'
st.markdown(f'<div class="step-indicator">{steps_html}</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-header">📄 YOUR RESUME</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume_upload")
    if resume_file:
        with st.spinner("Extracting..."):
            st.session_state.resume_text = extract_text_from_pdf(resume_file)
        st.markdown(f'<span class="status-pill success">✓ {len(st.session_state.resume_text.split())} words loaded</span>', unsafe_allow_html=True)
    else:
        st.session_state.resume_text = st.text_area("Or paste resume text", height=200, placeholder="Paste your resume here...")

with col2:
    st.markdown('<div class="card-header">💼 JOB DESCRIPTION</div>', unsafe_allow_html=True)
    jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], key="jd_upload")
    if jd_file:
        with st.spinner("Extracting..."):
            st.session_state.jd_text = extract_text_from_pdf(jd_file)
        st.markdown(f'<span class="status-pill success">✓ {len(st.session_state.jd_text.split())} words loaded</span>', unsafe_allow_html=True)
    else:
        st.session_state.jd_text = st.text_area("Or paste job description", height=200, placeholder="Paste the job description here...")

st.markdown("---")

# ── Analyze ───────────────────────────────────────────────────────────────────
can_analyze = bool(st.session_state.resume_text.strip() and st.session_state.jd_text.strip() and api_key and api_key.strip())

col_btn, col_hint = st.columns([2, 5])
with col_btn:
    analyze_clicked = st.button("🚀 ANALYZE APPLICATION", disabled=not can_analyze)
with col_hint:
    if not st.session_state.resume_text.strip() or not st.session_state.jd_text.strip():
        st.markdown('<span style="color:#6b6b8a;font-size:0.85rem;">↑ Upload or paste resume & job description</span>', unsafe_allow_html=True)
    elif not api_key:
        if "Groq" in llm_provider:
            st.markdown('<span style="color:#f59e0b;font-size:0.85rem;">↑ Add your free Groq key (console.groq.com/keys)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#f59e0b;font-size:0.85rem;">↑ Add your OpenAI key in the sidebar</span>', unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if analyze_clicked and can_analyze:
    st.session_state.step = 1
    st.session_state.results = {}

    with st.spinner("Building RAG pipeline & vectorizing documents..."):
        try:
            pipeline = RAGPipeline(
                resume_text=st.session_state.resume_text,
                jd_text=st.session_state.jd_text,
                api_key=api_key.strip(),
                provider=provider,
                model_name=model_name,
                vector_store_type=vector_store.lower()
            )
        except Exception as e:
            st.error(f"Pipeline init failed: {e}")
            st.stop()

    results = {}
    if run_skill_gap:
        with st.spinner("🔍 Analyzing skill gaps..."):
            try:
                results["skill_gap"] = pipeline.analyze(SKILL_GAP_PROMPT, task="skill_gap")
            except Exception as e:
                results["skill_gap"] = f"Error: {e}"

    if run_resume_rewrite:
        with st.spinner("✏️ Rewriting resume bullets..."):
            try:
                results["resume_rewrite"] = pipeline.analyze(RESUME_REWRITE_PROMPT, task="resume_rewrite")
            except Exception as e:
                results["resume_rewrite"] = f"Error: {e}"

    if run_cover_letter:
        with st.spinner("📝 Generating cover letter..."):
            try:
                results["cover_letter"] = pipeline.analyze(COVER_LETTER_PROMPT, task="cover_letter")
            except Exception as e:
                results["cover_letter"] = f"Error: {e}"

    st.session_state.results = results
    st.session_state.step = 2
    st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("---")
    st.markdown('<div class="card-header">✨ ANALYSIS RESULTS</div>', unsafe_allow_html=True)

    tab_labels = []
    if "skill_gap" in st.session_state.results: tab_labels.append("🔍 Skill Gap")
    if "resume_rewrite" in st.session_state.results: tab_labels.append("✏️ Resume Rewrites")
    if "cover_letter" in st.session_state.results: tab_labels.append("📝 Cover Letter")

    if tab_labels:
        tabs = st.tabs(tab_labels)
        idx = 0
        styles = ["", " cyan", " green"]
        keys = ["skill_gap", "resume_rewrite", "cover_letter"]
        fnames = ["skill_gap_analysis.txt", "resume_rewrites.txt", "cover_letter.txt"]

        for i, key in enumerate(keys):
            if key in st.session_state.results:
                with tabs[idx]:
                    st.markdown(f'<div class="result-box{styles[i]}">{st.session_state.results[key]}</div>', unsafe_allow_html=True)
                    st.download_button(f"⬇ Download", st.session_state.results[key], file_name=fnames[i], mime="text/plain")
                idx += 1

    st.markdown("---")
    if st.button("🔄 New Analysis"):
        st.session_state.results = {}
        st.session_state.step = 0
        st.rerun()
