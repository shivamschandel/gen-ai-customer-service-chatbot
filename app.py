"""
NEXUS – AI Customer Service Intelligence Platform
Powered by Groq (FREE) — LLaMA 3.3 70B
Tasks: Dynamic KB (1) · Multi-modal (2) · arXiv Research (4) · Sentiment (5) · Multi-language (6)
"""

import os, sys, datetime
from typing import Dict, List

import streamlit as st

st.set_page_config(
    page_title="NEXUS AI | Customer Service",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))

import config
from modules.knowledge_base import KnowledgeBase
from modules.sentiment       import SentimentAnalyzer
from modules.multilingual    import LanguageHandler, LANGUAGE_PROFILES
from modules.arxiv_expert    import ArxivExpert
from modules.multimodal      import MultimodalHandler

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg0:#070b14;--bg1:#0d1117;--bg2:#111827;--bg3:#1a2235;--cyan:#06b6d4;--purple:#8b5cf6;--green:#10b981;--amber:#f59e0b;--red:#ef4444;--text1:#f1f5f9;--text2:#94a3b8;--border:rgba(6,182,212,0.18);--glow:0 0 24px rgba(6,182,212,0.25);--grad:linear-gradient(135deg,#06b6d4,#8b5cf6)}
html,body,.stApp{background:var(--bg0)!important}
.stApp{font-family:'DM Sans',sans-serif;color:var(--text1)}
h1,h2,h3,h4{font-family:'Syne',sans-serif}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border)}
.nexus-logo{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-family:'Syne',sans-serif;font-weight:800;font-size:1.7rem;letter-spacing:.08em}
.nexus-sub{color:var(--text2);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;margin-top:2px}
.logo-wrap{padding:.6rem 0 1.2rem;border-bottom:1px solid var(--border);margin-bottom:.6rem}
.stat-row{display:flex;gap:10px;margin:10px 0}
.stat-card{flex:1;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:10px 12px;text-align:center}
.stat-val{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.stat-lbl{font-size:.68rem;color:var(--text2);text-transform:uppercase;letter-spacing:.05em}
.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;border-radius:20px;font-size:.68rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;border:1px solid}
.badge-positive{background:rgba(16,185,129,.12);color:#10b981;border-color:rgba(16,185,129,.35)}
.badge-negative{background:rgba(239,68,68,.12);color:#ef4444;border-color:rgba(239,68,68,.35)}
.badge-neutral{background:rgba(148,163,184,.12);color:#94a3b8;border-color:rgba(148,163,184,.35)}
.badge-mixed{background:rgba(245,158,11,.12);color:#f59e0b;border-color:rgba(245,158,11,.35)}
.paper-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:10px 0}
.paper-title{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:var(--text1);margin-bottom:4px}
.paper-meta{font-size:.75rem;color:var(--text2);margin-bottom:8px}
.paper-abstract{font-size:.82rem;color:var(--text2);line-height:1.55}
.mode-pill{display:inline-block;background:rgba(139,92,246,.15);border:1px solid rgba(139,92,246,.4);color:#a78bfa;border-radius:20px;padding:3px 12px;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.kb-card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:8px 0}
.stButton>button{background:var(--grad)!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;box-shadow:0 4px 14px rgba(6,182,212,.25)!important}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:var(--bg2)!important;color:var(--text1)!important;border:1px solid var(--border)!important;border-radius:8px!important}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{border-color:var(--cyan)!important;box-shadow:0 0 0 2px rgba(6,182,212,.2)!important}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2);border-radius:10px;padding:4px;gap:4px;border:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--text2)!important;border-radius:7px!important;font-size:.82rem!important;padding:6px 14px!important}
.stTabs [aria-selected="true"]{background:var(--grad)!important;color:white!important}
hr{border-color:var(--border)!important}
[data-testid="stChatMessage"]{background:var(--bg2)!important;border:1px solid var(--border);border-radius:12px!important;padding:12px!important;margin-bottom:8px}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg0)}
::-webkit-scrollbar-thumb{background:var(--bg3);border-radius:10px}
::-webkit-scrollbar-thumb:hover{background:var(--cyan)}
.free-badge{display:inline-block;background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.4);color:#10b981;border-radius:20px;padding:3px 12px;font-size:.75rem;font-weight:700}
.groq-badge{display:inline-block;background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.4);color:#f59e0b;border-radius:20px;padding:3px 12px;font-size:.75rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "messages": [], "api_key": config.GROQ_API_KEY, "page": "💬 Chat",
    "language_mode": "auto", "manual_lang": "en", "last_detected_lang": "en",
    "kb_use_context": True, "arxiv_mode": False, "arxiv_papers": [],
    "current_image": None, "sentiment_history": [],
    "kb_sources": config.DEFAULT_UPDATE_SOURCES[:],
    "update_interval": 24, "message_count": 0,
    "session_start": datetime.datetime.now().isoformat(),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# Module Init
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚡ Initialising NEXUS with Groq…")
def load_modules(api_key: str):
    kb = KnowledgeBase(persist_dir=config.CHROMA_PERSIST_DIR, collection_name=config.COLLECTION_NAME)
    sa = SentimentAnalyzer(api_key=api_key, model=config.FAST_MODEL)
    lh = LanguageHandler(api_key=api_key,   model=config.FAST_MODEL)
    ax = ArxivExpert(api_key=api_key,        model=config.MODEL)
    mm = MultimodalHandler(api_key=api_key,  model=config.VISION_MODEL)
    return kb, sa, lh, ax, mm

def get_modules():
    if not st.session_state.api_key: return None, None, None, None, None
    return load_modules(st.session_state.api_key)

# ══════════════════════════════════════════════════════════════════════════════
# Core Response Generator
# ══════════════════════════════════════════════════════════════════════════════
def generate_response(user_msg: str, img_b64: str = None, img_mime: str = None) -> Dict:
    from groq import Groq

    kb, sa, lh, ax, mm = get_modules()
    if not kb:
        return {"text": "⚠️ Please enter your Groq API key in ⚙️ Settings.", "meta": {}}

    client = Groq(api_key=st.session_state.api_key)
    meta   = {}

    # 1 — Language detection
    lang = lh.detect_language(user_msg) if st.session_state.language_mode == "auto" else st.session_state.manual_lang
    st.session_state.last_detected_lang = lang
    lp   = lh.get_profile(lang)
    meta.update({"lang": lang, "lang_name": lp["name"], "lang_flag": lp["flag"]})

    # 2 — Sentiment
    sentiment = sa.analyze(user_msg)
    tone      = sa.get_response_tone(sentiment)
    meta["sentiment"] = sentiment
    st.session_state.sentiment_history.append(sentiment)

    # 3 — KB context
    kb_context = ""
    if st.session_state.kb_use_context:
        hits = kb.search(user_msg, n_results=3)
        if hits:
            kb_context = "\n\n## Knowledge Base Context\n"
            for i, h in enumerate(hits, 1):
                kb_context += f"[{i}] (score {h['score']:.2f}): {h['content']}\n\n"
            meta["kb_hits"] = len(hits)

    # 4 — arXiv context
    arxiv_ctx = ""
    if st.session_state.arxiv_mode and st.session_state.arxiv_papers:
        arxiv_ctx = "\n\n## Active Research Papers\n"
        for p in st.session_state.arxiv_papers[:3]:
            arxiv_ctx += f"**{p['title']}** ({p.get('published','?')})\n{p['abstract'][:250]}…\n\n"

    # 5 — Build system prompt
    system = config.BASE_SYSTEM_PROMPT
    system += lh.build_language_instruction(lang)
    system += f"\n\n## Tone Instruction\n{tone}"
    if kb_context: system += kb_context
    if arxiv_ctx:  system += "\n\n## Research Mode\n" + arxiv_ctx

    # 6 — Build message history
    messages = [{"role": "system", "content": system}]
    for m in st.session_state.messages[-16:]:
        role    = "user" if m["role"] == "user" else "assistant"
        content = m["content"] if isinstance(m["content"], str) else next(
            (p["text"] for p in m["content"] if p.get("type") == "text"), ""
        )
        messages.append({"role": role, "content": content})

    # 7 — Current user message (with optional image)
    if img_b64 and img_mime:
        messages.append({
            "role": "user",
            "content": mm.build_vision_content(user_msg, img_b64, img_mime),
        })
        model_to_use = config.VISION_MODEL
    else:
        messages.append({"role": "user", "content": user_msg})
        model_to_use = config.MODEL

    # 8 — Call Groq
    try:
        resp  = client.chat.completions.create(
            model=model_to_use,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
        )
        reply = resp.choices[0].message.content
        meta["tokens_used"] = resp.usage.total_tokens if resp.usage else "—"
    except Exception as e:
        reply = f"⚠️ Groq API error: {e}"

    return {"text": reply, "meta": meta}

# ══════════════════════════════════════════════════════════════════════════════
# UI Helpers
# ══════════════════════════════════════════════════════════════════════════════
def sentiment_badge(s: Dict) -> str:
    snt = s.get("sentiment", "neutral")
    return f'<span class="badge badge-{snt}">{config.SENTIMENT_ICONS.get(snt,"😐")} {snt}</span>'

def lang_badge(flag: str, name: str) -> str:
    return f'<span class="badge badge-neutral">{flag} {name}</span>'

def render_meta(meta: Dict):
    parts = []
    if "sentiment" in meta: parts.append(sentiment_badge(meta["sentiment"]))
    if "lang_flag"  in meta: parts.append(lang_badge(meta["lang_flag"], meta.get("lang_name", "")))
    if "kb_hits"    in meta: parts.append(f'<span class="badge badge-neutral">📚 {meta["kb_hits"]} KB hits</span>')
    if "tokens_used" in meta: parts.append(f'<span class="badge badge-neutral">⚡ {meta["tokens_used"]} tokens</span>')
    if parts: st.markdown(" ".join(parts), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ══════════════════════════════════════════════════════════════════════════════
def page_chat():
    st.markdown("""
    <div style='margin-bottom:16px'>
      <h2 style='font-family:Syne,sans-serif;margin:0;font-size:1.5rem'>💬 Customer Support Chat</h2>
      <p style='color:#94a3b8;font-size:.82rem;margin:4px 0 0'>
        Multi-lingual · Sentiment-aware · Knowledge-enhanced · Vision-capable &nbsp;
        <span class="free-badge">✅ FREE</span> &nbsp;
        <span class="groq-badge">⚡ Groq LLaMA 3.3</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Active mode pills
    modes = []
    if st.session_state.kb_use_context: modes.append("📚 KB Context")
    if st.session_state.arxiv_mode:     modes.append("🔬 Research Mode")
    flag  = LANGUAGE_PROFILES.get(st.session_state.last_detected_lang, {}).get("flag", "🌐")
    modes.append(f"{flag} {'Auto-Lang' if st.session_state.language_mode=='auto' else LANGUAGE_PROFILES.get(st.session_state.manual_lang,{}).get('name','?')}")
    st.markdown("  ".join(f'<span class="mode-pill">{m}</span>' for m in modes), unsafe_allow_html=True)
    st.markdown("")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            content = msg["content"]
            if isinstance(content, list):
                for p in content:
                    if p.get("type") == "text": st.markdown(p["text"])
            else:
                st.markdown(content)
            if "img_b64" in msg:
                st.image(f"data:{msg.get('img_mime','image/png')};base64,{msg['img_b64']}", width=280)
            if "meta" in msg and msg["role"] == "assistant":
                render_meta(msg["meta"])

    # Image upload
    with st.expander("📎 Attach an image", expanded=False):
        uploaded = st.file_uploader("Upload screenshot or photo",
                                    type=["jpg","jpeg","png","webp"], key="img_upload")
        if uploaded:
            st.image(uploaded, caption="Preview", width=260)
            try:
                mm = get_modules()[4]
                b64, mime = mm.process_image(uploaded)
                st.session_state.current_image = (b64, mime, uploaded.name)
                st.success(f"✅ Ready: {uploaded.name}")
            except Exception as e:
                st.error(str(e))
        else:
            st.session_state.current_image = None

    # Chat input
    user_input = st.chat_input("Type your message…")
    if user_input:
        if not st.session_state.api_key:
            st.error("⚠️ Enter your Groq API key in ⚙️ Settings first."); return

        img_b64, img_mime, img_name = None, None, None
        if st.session_state.current_image:
            img_b64, img_mime, img_name = st.session_state.current_image

        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
            if img_b64:
                st.image(f"data:{img_mime};base64,{img_b64}", caption=f"📎 {img_name}", width=250)

        user_obj = {"role": "user", "content": user_input}
        if img_b64: user_obj.update({"img_b64": img_b64, "img_mime": img_mime})
        st.session_state.messages.append(user_obj)
        st.session_state.message_count += 1

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⚡ NEXUS thinking (Groq)…"):
                result = generate_response(user_input, img_b64, img_mime)
            st.markdown(result["text"])
            render_meta(result["meta"])

        st.session_state.messages.append({
            "role": "assistant", "content": result["text"], "meta": result["meta"]
        })
        st.session_state.current_image = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Page: Knowledge Base
# ══════════════════════════════════════════════════════════════════════════════
def page_knowledge_base():
    st.markdown("<h2 style='font-family:Syne,sans-serif'>📚 Knowledge Base</h2>", unsafe_allow_html=True)
    kb, *_ = get_modules()
    if not kb: st.warning("Enter your API key in Settings."); return

    stats = kb.get_stats()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🗂️ Chunks",     stats["total_chunks"])
    c2.metric("🌐 Sources",    stats["sources"])
    c3.metric("⏱️ Scheduler",  "✅ On" if stats["scheduler_running"] else "⏸ Off")
    c4.metric("📅 Log Entries",len(stats["recent_updates"]))
    st.markdown("---")

    tabs = st.tabs(["✍️ Add Text","🌐 Add URL","🗓️ Schedule","📋 Log","⚠️ Clear"])

    with tabs[0]:
        title  = st.text_input("Title", placeholder="e.g. Return Policy")
        body   = st.text_area("Content", height=160, placeholder="Paste FAQs, policies…")
        source = st.text_input("Source (optional)")
        if st.button("➕ Add to KB"):
            if body.strip():
                with st.spinner("Embedding…"):
                    res = kb.add_document(body, metadata={"title": title, "source": source or "manual"})
                st.success(f"✅ Added {res['chunks']} chunks") if res["success"] else st.error(res["error"])
            else: st.warning("Enter content first.")

    with tabs[1]:
        url = st.text_input("URL", placeholder="https://example.com/help")
        if st.button("🌐 Scrape & Ingest"):
            if url.startswith("http"):
                with st.spinner("Scraping…"):
                    res = kb.update_from_url(url)
                st.success(f"✅ {res['chunks']} chunks ingested") if res["success"] else st.error(res.get("error"))
            else: st.warning("Enter a valid URL.")

    with tabs[2]:
        src_text = st.text_area("URLs (one per line)", value="\n".join(st.session_state.kb_sources), height=120)
        interval = st.slider("Interval (hours)", 1, 168, st.session_state.update_interval)
        col1,col2 = st.columns(2)
        with col1:
            if st.button("▶ Start"):
                urls = [u.strip() for u in src_text.splitlines() if u.strip().startswith("http")]
                if urls:
                    st.session_state.kb_sources = urls; st.session_state.update_interval = interval
                    kb.schedule_updates(urls, interval_hours=interval)
                    st.success(f"✅ Scheduler started — {len(urls)} source(s) every {interval}h")
        with col2:
            if st.button("⏹ Stop"): kb.stop_scheduler(); st.success("Stopped.")

    with tabs[3]:
        updates = stats["recent_updates"]
        if updates:
            for u in reversed(updates[-8:]):
                icon = "✅" if u["success"] else "❌"
                st.markdown(f'<div class="kb-card"><b>{icon} {u.get("url","manual")}</b><br><span style="color:#94a3b8;font-size:.8rem">{u.get("timestamp","")} | {u.get("chunks",0)} chunks | {u.get("error") or "OK"}</span></div>', unsafe_allow_html=True)
        else: st.info("No updates logged yet.")

    with tabs[4]:
        st.warning("This permanently deletes ALL documents.")
        confirm = st.text_input("Type DELETE to confirm")
        if st.button("🗑️ Clear All"):
            if confirm == "DELETE": kb.delete_all(); st.success("Cleared."); st.rerun()
            else: st.error("Type DELETE exactly.")

# ══════════════════════════════════════════════════════════════════════════════
# Page: Research
# ══════════════════════════════════════════════════════════════════════════════
def page_research():
    st.markdown("<h2 style='font-family:Syne,sans-serif'>🔬 Research Expert</h2>", unsafe_allow_html=True)
    _,_,_,ax,_ = get_modules()
    if not ax: st.warning("Enter your API key in Settings."); return
    if not ax.available: st.error("Install arxiv: `pip install arxiv`"); return

    tabs = st.tabs(["🔍 Search","💡 Explain","❓ Q&A","📊 Analytics"])

    with tabs[0]:
        col1,col2,col3 = st.columns([3,1,1])
        with col1: query = st.text_input("Search query", placeholder="e.g. large language models")
        with col2: cat_label = st.selectbox("Category", ["Any"] + list(config.ARXIV_CATEGORIES.values()))
        with col3: n = st.number_input("# Results", 3, 12, 6)
        cat_code = next((k for k,v in config.ARXIV_CATEGORIES.items() if v == cat_label), "")
        if st.button("🔍 Search arXiv"):
            if query:
                with st.spinner("Searching arXiv…"):
                    papers = ax.search_papers(query, max_results=n, category=cat_code if cat_code else None)
                st.session_state.arxiv_papers = papers
                st.success(f"Found {len(papers)} paper(s)")
        if st.session_state.arxiv_papers:
            st.session_state.arxiv_mode = st.toggle("🧑‍🔬 Research Mode in Chat", value=st.session_state.arxiv_mode)
        for paper in st.session_state.arxiv_papers:
            if "error" in paper: st.error(paper.get("abstract","")); continue
            with st.expander(f"📄 {paper['title'][:90]}"):
                st.markdown(f'<div class="paper-card"><div class="paper-title">{paper["title"]}</div><div class="paper-meta">👥 {", ".join(paper.get("authors",[])[:3])} | 📅 {paper.get("published","?")} | 🏷️ {", ".join(paper.get("categories",[])[:3])}</div><div class="paper-abstract">{paper.get("abstract","")[:400]}…</div></div>', unsafe_allow_html=True)
                cA,cB = st.columns(2)
                with cA:
                    if st.button("📝 Summarize", key=f"s_{paper['id']}"):
                        with st.spinner("Summarising…"):
                            st.markdown(ax.summarize_paper(paper))
                with cB:
                    if paper.get("pdf_url"): st.markdown(f"[📥 PDF]({paper['pdf_url']})")

    with tabs[1]:
        concept = st.text_input("Concept", placeholder="e.g. Transformer attention mechanism")
        use_p   = st.checkbox("Use loaded papers as context", value=bool(st.session_state.arxiv_papers))
        if st.button("💡 Explain"):
            if concept:
                with st.spinner("Generating explanation…"):
                    st.markdown(ax.explain_concept(concept, st.session_state.arxiv_papers if use_p else None))

    with tabs[2]:
        if not st.session_state.arxiv_papers: st.info("Search papers first.")
        else:
            q = st.text_area("Research question", height=90)
            if st.button("🤔 Answer"):
                if q:
                    with st.spinner("Synthesising answer…"):
                        st.markdown(ax.answer_with_papers(q, st.session_state.arxiv_papers))

    with tabs[3]:
        if st.session_state.arxiv_papers:
            import plotly.express as px, pandas as pd
            viz = ax.get_visualization_data(st.session_state.arxiv_papers)
            if viz["category_counts"]:
                df  = pd.DataFrame(list(viz["category_counts"].items()), columns=["Category","Count"])
                fig = px.pie(df, names="Category", values="Count", hole=0.45,
                             title="Paper Categories", color_discrete_sequence=px.colors.sequential.Plasma_r)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9")
                st.plotly_chart(fig, use_container_width=True)
        else: st.info("Search papers to see analytics.")

# ══════════════════════════════════════════════════════════════════════════════
# Page: Analytics
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("<h2 style='font-family:Syne,sans-serif'>📊 Sentiment Analytics</h2>", unsafe_allow_html=True)
    import plotly.express as px, plotly.graph_objects as go, pandas as pd

    history = st.session_state.sentiment_history
    if not history: st.info("No data yet — start chatting!"); return

    breakdown: Dict[str,int] = {}
    urgent = esc = 0
    for h in history:
        s = h.get("sentiment","neutral"); breakdown[s] = breakdown.get(s,0)+1
        if h.get("urgency")=="high":         urgent += 1
        if h.get("requires_escalation"):     esc    += 1

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💬 Total",    len(history))
    c2.metric("😊 Positive", breakdown.get("positive",0))
    c3.metric("😟 Negative", breakdown.get("negative",0))
    c4.metric("🚨 Urgent",   urgent)
    c5.metric("🆘 Escalations", esc)
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        cmap = {"positive":"#10b981","negative":"#ef4444","neutral":"#94a3b8","mixed":"#f59e0b"}
        df   = pd.DataFrame(list(breakdown.items()), columns=["Sentiment","Count"])
        fig  = px.pie(df, names="Sentiment", values="Count", hole=0.5, title="Sentiment Breakdown",
                      color="Sentiment", color_discrete_map=cmap)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#f1f5f9", legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        order = {"positive":3,"neutral":2,"mixed":1,"negative":0}
        df2   = pd.DataFrame([{"#":i+1,"score":order.get(h.get("sentiment","neutral"),2),
                                "sentiment":h.get("sentiment","neutral")} for i,h in enumerate(history)])
        fig2  = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df2["#"], y=df2["score"], mode="lines+markers",
            line=dict(color="#06b6d4", width=2.5),
            marker=dict(size=8, color=df2["score"],
                        colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]]),
            text=df2["sentiment"], hovertemplate="Msg %{x}<br>%{text}<extra></extra>",
        ))
        fig2.update_layout(
            title="Sentiment Timeline", xaxis_title="Message #",
            yaxis=dict(tickvals=[0,1,2,3], ticktext=["Negative","Mixed","Neutral","Positive"]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"), yaxis_gridcolor="rgba(255,255,255,0.06)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    rows = [{"#":i+1,
             "Sentiment": f'{config.SENTIMENT_ICONS.get(h.get("sentiment","neutral"),"😐")} {h.get("sentiment","?")}',
             "Intensity": h.get("intensity","?"), "Urgency": h.get("urgency","?"),
             "Confidence": f'{h.get("confidence",0):.0%}',
             "Escalation": "⚠️ Yes" if h.get("requires_escalation") else "No",
             "Emotions": ", ".join(h.get("emotions",[])[:3]) or "—"} for i,h in enumerate(history)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear History"): st.session_state.sentiment_history=[]; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Page: Settings
# ══════════════════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown("<h2 style='font-family:Syne,sans-serif'>⚙️ Settings</h2>", unsafe_allow_html=True)
    tabs = st.tabs(["🔑 API Key","🌐 Language","📚 KB","ℹ️ About"])

    with tabs[0]:
        st.markdown("#### Groq API Key")
        st.markdown('<span class="free-badge">✅ 100% FREE</span> &nbsp; <span class="groq-badge">⚡ LLaMA 3.3 70B</span>', unsafe_allow_html=True)
        st.markdown("""
**How to get your FREE Groq API key (takes 1 minute):**
1. Go to 👉 **[console.groq.com](https://console.groq.com)**
2. Sign up with **email or Google** — no credit card needed
3. Click **API Keys** → **Create API Key**
4. Copy and paste it below ⬇️
        """)
        key = st.text_input("Groq API Key", value=st.session_state.api_key,
                            type="password", placeholder="gsk_…")
        if st.button("💾 Save API Key"):
            st.session_state.api_key = key
            load_modules.clear()
            st.success("✅ API key saved! Modules will reinitialise.")

        st.markdown(f"""---
| Model | Usage |
|---|---|
| `{config.MODEL}` | Main chat |
| `{config.FAST_MODEL}` | Sentiment & language |
| `{config.VISION_MODEL}` | Image understanding |

**Free limits:** 14,400 requests/day · 500,000 tokens/day
        """)

    with tabs[1]:
        mode = st.radio("Detection mode", ["auto","manual"],
                        format_func=lambda x: "🌐 Auto-detect" if x=="auto" else "✋ Manual select",
                        index=0 if st.session_state.language_mode=="auto" else 1)
        st.session_state.language_mode = mode
        if mode == "manual":
            options = list(LANGUAGE_PROFILES.keys())
            idx     = options.index(st.session_state.manual_lang) if st.session_state.manual_lang in options else 0
            chosen  = st.selectbox("Language", options, index=idx,
                                   format_func=lambda c: f"{LANGUAGE_PROFILES[c]['flag']} {LANGUAGE_PROFILES[c]['name']}")
            st.session_state.manual_lang = chosen
        st.markdown("---\n**Supported Languages:**")
        cols = st.columns(4)
        for i,(code,info) in enumerate(LANGUAGE_PROFILES.items()):
            cols[i%4].markdown(f"{info['flag']} {info['name']}")

    with tabs[2]:
        st.session_state.kb_use_context = st.toggle(
            "Inject KB context into every chat", value=st.session_state.kb_use_context
        )

    with tabs[3]:
        st.markdown("""
## NEXUS AI — Powered by Groq (FREE ⚡)

| Task | Feature | Status |
|------|---------|--------|
| 1 | Dynamic Knowledge Base (ChromaDB + web scraping) | ✅ |
| 2 | Multi-modal vision (LLaMA 4 Scout) | ✅ |
| 4 | arXiv Research Expert | ✅ |
| 5 | Sentiment Analysis (LLaMA 3.1 8B) | ✅ |
| 6 | 10-language auto-detection | ✅ |

**AI Provider:** Groq Cloud — fastest free LLM inference  
**Models:** LLaMA 3.3 70B · LLaMA 3.1 8B · LLaMA 4 Scout (vision)  
**Free limits:** 14,400 req/day · No credit card required  
**Stack:** Streamlit · Groq · ChromaDB · langdetect · arxiv · plotly
        """)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="logo-wrap">
          <div class="nexus-logo">NEXUS</div>
          <div class="nexus-sub">AI Customer Service · Groq ⚡</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Navigation")
        pages = ["💬 Chat","📚 Knowledge Base","🔬 Research","📊 Analytics","⚙️ Settings"]
        page  = st.radio("", pages, index=pages.index(st.session_state.page), label_visibility="collapsed")
        st.session_state.page = page
        st.markdown("---")

        st.markdown("### Session Stats")
        msgs = st.session_state.message_count
        lang = st.session_state.last_detected_lang
        flag = LANGUAGE_PROFILES.get(lang, {}).get("flag", "🌐")
        st.markdown(f'<div class="stat-row"><div class="stat-card"><div class="stat-val">{msgs}</div><div class="stat-lbl">Messages</div></div><div class="stat-card"><div class="stat-val">{flag}</div><div class="stat-lbl">{LANGUAGE_PROFILES.get(lang,{}).get("name","?")}</div></div></div>', unsafe_allow_html=True)

        history = st.session_state.sentiment_history
        if history:
            last = history[-1]; snt = last.get("sentiment","neutral")
            st.markdown(f'<div class="stat-card" style="margin-top:8px"><div class="stat-val">{config.SENTIMENT_ICONS.get(snt,"😐")}</div><div class="stat-lbl">Last: {snt}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Quick Controls")
        st.session_state.kb_use_context = st.toggle("📚 KB Context",    value=st.session_state.kb_use_context)
        st.session_state.arxiv_mode     = st.toggle("🔬 Research Mode", value=st.session_state.arxiv_mode)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []; st.session_state.message_count = 0; st.rerun()
        st.markdown("---")
        st.markdown('<div style="color:#475569;font-size:.7rem;text-align:center">NEXUS v3.0 · Groq LLaMA 3.3 (Free)</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    if not st.session_state.api_key and st.session_state.page != "⚙️ Settings":
        st.markdown("""
        <div style='text-align:center;padding:60px 20px'>
          <div class='nexus-logo' style='font-size:3rem'>NEXUS</div>
          <p style='color:#94a3b8;margin:12px 0 24px'>AI Customer Service Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("👉 Go to **⚙️ Settings** → enter your **free Groq API key** → start chatting!")
        st.markdown("**Get your free key at:** 👉 [console.groq.com](https://console.groq.com) — takes 1 minute, no credit card!")
        page_settings()
        return

    page = st.session_state.page
    if   page == "💬 Chat":           page_chat()
    elif page == "📚 Knowledge Base": page_knowledge_base()
    elif page == "🔬 Research":       page_research()
    elif page == "📊 Analytics":      page_analytics()
    elif page == "⚙️ Settings":       page_settings()

if __name__ == "__main__":
    main()
