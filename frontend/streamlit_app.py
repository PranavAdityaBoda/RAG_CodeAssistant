"""
CodeLens frontend.
Ingest, document, query and contribute to any GitHub repository
"""
import io
import re as _re
import time

import pandas as pd
import streamlit as st

from api_client import (
    BackendError,
    ask_question,
    check_backend_health,
    create_pull_request,
    get_github_login_url,
    get_github_status,
    get_ingest_result,
    get_ingest_status,
    get_job_status,
    get_llm_usage,
    ingest_repository,
    poll_doc_job,
    start_doc_generation,
)

st.set_page_config(
    page_title="CodeLens",
    page_icon="",
    layout="wide",
    menu_items={},      
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='margin-bottom:0;'>CodeLens</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:1.1rem;color:#aaaaaa;margin-top:0.2rem;margin-bottom:1.2rem;'>"
    "RAG-powered code documentation and GitHub integration using Groq LLM"
    "</p>",
    unsafe_allow_html=True,
)

# ── backend health check ──────────────────────────────────────────────────
if not check_backend_health():
    st.error(
        "Cannot reach the backend. Make sure it is running "
        "(`uvicorn app.main:app --reload` from the backend/ folder) "
        "or check the BACKEND_URL environment variable."
    )
    st.stop()

# ── OAuth token capture ───────────────────────────────────────────────────
params = st.query_params
if "github_token" in params and not st.session_state.get("github_token"):
    st.session_state["github_token"] = params["github_token"]
    st.query_params.clear()
    pre_job_id = st.session_state.pop("pre_oauth_job_id", None)
    if pre_job_id and not st.session_state.get("last_result"):
        try:
            job_status = get_job_status(pre_job_id)
            if job_status.get("status") == "done":
                st.session_state["last_result"] = {
                    "job_id":   pre_job_id,
                    "repo_url": job_status.get("repo_url", ""),
                    "files":    [],
                    "chunks":   [],
                    "_restored_after_oauth": True,
                }
        except Exception:
            pass
    st.rerun()

if "github_error" in params:
    st.error(f"GitHub login failed: {params['github_error']}")
    st.query_params.clear()

# ── sidebar: LLM usage only ───────────────────────────────────────────────
with st.sidebar:
    st.header("Usage")
    usage = get_llm_usage()
    if usage and "calls_today" in usage:
        st.metric("Requests today",  f"{usage['calls_today']} / {usage['daily_limit']}")
        st.metric("This minute",     f"{usage['calls_this_minute']} / {usage['rpm_limit']}")
        pct = usage["calls_today"] / usage["daily_limit"]
        st.progress(min(pct, 1.0), text="Daily budget")
    else:
        st.caption("Usage unavailable. Check GROQ_API_KEY.")

# ── tabs ──────────────────────────────────────────────────────────────────
st.markdown(
    "<style>div[data-testid='stTabs'] button {padding: 0.5rem 1.5rem;}</style>",
    unsafe_allow_html=True,
)
tab_ingest, tab_docs, tab_qa, tab_pr, tab_info = st.tabs(
    ["  Ingest  ", "  Generate Docs  ", "  RAG Code Assistant  ", "  Create PR  ", "  About  "]
)



# ─────────────────────────────────────────────────────────────────────────
# ABOUT SECTION
# ─────────────────────────────────────────────────────────────────────────
with tab_info:
    st.subheader("What is CodeLens?")
    st.markdown(
        "CodeLens is a RAG-based (Retrieval-Augmented Generation) developer tool "
        "that lets you ingest any GitHub repository, generate professional documentation, "
        "query the codebase in natural language and open pull requests, all from one UI."
    )

    st.divider()

    st.markdown("#### Why RAG?")
    st.markdown(
        "Large language models have a knowledge cutoff and no access to your private or "
        "recent code. RAG solves this by retrieving the most relevant chunks of your actual "
        "source code and passing them to the LLM as context, so every answer is grounded "
        "in the real codebase, not a hallucinated guess."
    )
    st.markdown(
        "CodeLens uses **Chroma** as the local vector database and "
        "**sentence-transformers** (`all-MiniLM-L6-v2`) for embeddings. "
        "both run entirely on your machine with no external API calls. "
        "Retrieval uses query expansion (generating alternative phrasings of your question) "
        "and deduplication to surface the most relevant chunks even when your wording "
        "doesn't match the code's vocabulary."
    )

    st.divider()

    # What it does
    st.markdown("#### What it does")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ingest**")
        st.markdown(
            "Clones any public GitHub repo, walks the file tree, chunks source files "
            "using tree-sitter AST parsing (whole functions and classes with exact line ranges), "
            "embeds each chunk and stores it in Chroma. Runs as a background task. "
            "the UI stays responsive while large repos are processed."
        )
        st.markdown("**Generate Docs**")
        st.markdown(
            "Uses a two-tier Groq LLM pipeline: Llama 3.1 8B summarises each file, "
            "Llama 3.3 70B assembles the final document. Produces three doc types: "
            "API Reference, README and Beginner Guide, downloadable as Markdown or PDF."
        )
    with col2:
        st.markdown("**RAG Code Assistant**")
        st.markdown(
            "RAG-powered chat grounded in the actual source code. Ask any question about "
            "the repository: how a function works, what a module does, how to extend it. "
            "and get answers with source file citations. Code suggestions can be saved "
            "directly to a pull request."
        )
        st.markdown("**Create PR**")
        st.markdown(
            "Connect your GitHub account via OAuth. The agent auto-forks public repos "
            "you don't own, creates a branch, commits all selected docs and code changes "
            "in one PR, and writes the PR description with the LLM. "
            "PRs are opened under your own GitHub account."
        )

    st.divider()

    st.markdown("#### Tech stack")
    st.markdown(
        "| Layer | Technology |\n"
        "|-------|------------|\n"
        "| LLM | Groq: Llama 3.1 8B (fast) + Llama 3.3 70B (reasoning) |\n"
        "| RAG | LangChain, Chroma, sentence-transformers |\n"
        "| Code parsing | tree-sitter AST chunking |\n"
        "| Backend | FastAPI, PyGithub, GitPython |\n"
        "| Frontend | Streamlit |\n"
        "| Infra | Docker Compose, Railway / Render |"
    )
    st.caption("Zero paid LLM spend. Everything runs on Groq's free tier.")


# ─────────────────────────────────────────────────────────────────────────
# INGEST SECTION
# ─────────────────────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Ingest a GitHub repository")
    st.caption(
        "Clones the repo, walks the file tree, chunks the code with tree-sitter, "
        "and stores embeddings in a local Chroma vector database."
    )

    with st.form("ingest_form"):
        repo_url = st.text_input(
            "GitHub repository URL",
            placeholder="https://github.com/psf/requests",
        )
        submitted = st.form_submit_button("Ingest repository", type="primary")

    if submitted:
        if not repo_url.strip():
            st.warning("Enter a repository URL first.")
        else:
            try:
                started = ingest_repository(repo_url.strip())
                st.session_state["ingest_job_id"]  = started["job_id"]
                st.session_state["ingest_repo_url"] = repo_url.strip()
                st.session_state.pop("last_result", None)
                st.session_state.pop("doc_job_id", None)
                st.session_state.pop("last_doc", None)
                st.session_state.pop("doc_history", None)
                st.session_state["chat_history"] = []
                st.session_state.pop("last_qa", None)
                st.rerun()
            except BackendError as exc:
                st.error(f"Ingestion failed to start: {exc}")

    # ── poll ingestion status ─────────────────────────────────────────────
    ingest_job_id = st.session_state.get("ingest_job_id")
    if ingest_job_id and not st.session_state.get("last_result"):
        try:
            status = get_ingest_status(ingest_job_id)
        except BackendError as exc:
            st.error(f"Status check failed: {exc}")
            status = None

        if status:
            s = status["status"]
            if s in ("queued", "cloning", "walking", "chunking", "embedding"):
                step_labels = {
                    "queued":    "Queued...",
                    "cloning":   "Cloning repository...",
                    "walking":   "Walking file tree...",
                    "chunking":  "Chunking source files...",
                    "embedding": "Generating embeddings...",
                }
                st.info(step_labels.get(s, s))
                time.sleep(3)
                st.rerun()
            elif s == "failed":
                st.error(f"Ingestion failed: {status.get('error', 'unknown error')}")
                st.session_state.pop("ingest_job_id", None)
            elif s == "done":
                try:
                    result = get_ingest_result(ingest_job_id)
                    st.session_state["last_result"] = result
                    st.session_state.pop("ingest_job_id", None)
                    st.rerun()
                except BackendError as exc:
                    st.error(f"Could not fetch result: {exc}")

    result = st.session_state.get("last_result")
    if result:
        st.success(f"Ingested **{result['repo_url']}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Files discovered", len(result["files"]))
        col2.metric("Chunks created",   len(result["chunks"]))
        col3.metric("Languages",        len({f["language"] for f in result["files"]}))

        file_tab, chunk_tab = st.tabs(["Files", "Chunks"])
        with file_tab:
            st.dataframe(pd.DataFrame(result["files"]), use_container_width=True, hide_index=True)
        with chunk_tab:
            chunk_type_filter = st.multiselect(
                "Filter by chunk type",
                options=sorted({c["chunk_type"] for c in result["chunks"]}),
            )
            chunks = result["chunks"]
            if chunk_type_filter:
                chunks = [c for c in chunks if c["chunk_type"] in chunk_type_filter]
            st.dataframe(pd.DataFrame(chunks), use_container_width=True, hide_index=True)
            if chunks:
                options = [
                    f"{c['file_path']} ({c['symbol_name'] or c['chunk_type']})"
                    for c in chunks
                ]
                idx = st.selectbox(
                    "Preview a chunk", range(len(options)),
                    format_func=lambda i: options[i],
                )
                st.code(chunks[idx]["preview"], language=chunks[idx]["language"])
    else:
        st.info("Enter a public GitHub repository URL above to get started.")


# ─────────────────────────────────────────────────────────────────────────
# GENERATE DOCS SECTION
# ─────────────────────────────────────────────────────────────────────────
def _make_pdf(markdown_text: str, doc_title: str = "Documentation") -> bytes:
    """
    Converts Markdown to a professionally formatted PDF.
    """
    try:
        import re
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            HRFlowable, Paragraph, Preformatted,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        PAGE_W, PAGE_H = A4
        MARGIN = 22 * mm
        NAVY  = colors.HexColor("#1F3864")
        DGREY = colors.HexColor("#333333")
        LGREY = colors.HexColor("#f4f4f4")
        MGREY = colors.HexColor("#888888")
        CODE_C = colors.HexColor("#2d6a4f")
        FN, FB, FM = "Helvetica", "Helvetica-Bold", "Courier"

        def s(name, **kw): return ParagraphStyle(name, **kw)
        sH1    = s("H1",     fontName=FB, fontSize=22, textColor=NAVY,  spaceAfter=5*mm, spaceBefore=8*mm,  leading=28)
        sH2    = s("H2",     fontName=FB, fontSize=15, textColor=NAVY,  spaceAfter=3*mm, spaceBefore=6*mm,  leading=20)
        sH3    = s("H3",     fontName=FB, fontSize=12, textColor=DGREY, spaceAfter=2*mm, spaceBefore=4*mm,  leading=16)
        sBody  = s("Body",   fontName=FN, fontSize=10, textColor=DGREY, spaceAfter=2.5*mm, leading=15)
        sBullet= s("Bullet", fontName=FN, fontSize=10, textColor=DGREY, spaceAfter=1.5*mm, leftIndent=10*mm, leading=14)
        sCode  = s("Code",   fontName=FM, fontSize=8.5, textColor=CODE_C, backColor=LGREY,
                   spaceAfter=3*mm, spaceBefore=2*mm, leftIndent=6*mm, rightIndent=6*mm, leading=12)
        sQuote = s("Quote",  fontName=FN, fontSize=10, textColor=MGREY, spaceAfter=2*mm, leftIndent=8*mm, leading=14)

        def on_page(canvas, doc):
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#cccccc"))
            canvas.line(MARGIN, MARGIN-4*mm, PAGE_W-MARGIN, MARGIN-4*mm)
            canvas.setFont(FN, 8); canvas.setFillColor(MGREY)
            canvas.drawCentredString(PAGE_W/2, MARGIN-7*mm, f"Page {doc.page}")
            canvas.restoreState()

        def inline(text):
            text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
            text = re.sub(r"`([^`]+)`",
                lambda m: f'<font name="Courier" color="#2d6a4f">{m.group(1)}</font>', text)
            return text

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=MARGIN, bottomMargin=MARGIN+4*mm,
                                title=doc_title, author="CodeLens")

        import re as _re2
        markdown_text = _re2.sub(
            r"(^# .+\n)([^\n#\-].+\n?)*",
            lambda m: m.group(1),
            markdown_text,
            count=1,
            flags=_re2.MULTILINE,
        )

        story, lines = [], markdown_text.splitlines()
        i, in_code, code_lines = 0, False, []

        while i < len(lines):
            raw = lines[i]; stripped = raw.strip()

            if stripped.startswith("```"):
                if not in_code:
                    in_code = True; code_lines = []
                else:
                    in_code = False
                    story.append(Preformatted("\n".join(code_lines), sCode))
                i += 1; continue
            if in_code:
                code_lines.append(raw); i += 1; continue

            if re.match(r"^-{3,}$", stripped) or stripped in ("---","***","___"):
                story.append(HRFlowable(width="100%", thickness=0.5,
                    color=colors.HexColor("#cccccc"), spaceAfter=4*mm, spaceBefore=4*mm))
                i += 1; continue

            if stripped.startswith("# "):
                story.append(Paragraph(inline(stripped[2:]), sH1)); i += 1; continue
            if stripped.startswith("## "):
                story.append(Paragraph(inline(stripped[3:]), sH2)); i += 1; continue
            if stripped.startswith("### "):
                story.append(Paragraph(inline(stripped[4:]), sH3)); i += 1; continue

            if stripped.startswith("|") and stripped.endswith("|"):
                trows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row = lines[i].strip()
                    if re.match(r"^\|[\s\-:|]+\|$", row): i += 1; continue
                    trows.append([c.strip() for c in row.strip("|").split("|")]); i += 1
                if trows:
                    nc = max(len(r) for r in trows)
                    cw = (PAGE_W-2*MARGIN)/nc
                    sHdr = ParagraphStyle("TblHdr", fontName=FB, fontSize=9,
                                          textColor=colors.white)
                    td = []
                    for ri, row in enumerate(trows):
                        st_ = sHdr if ri == 0 else sBody
                        td.append([Paragraph(inline(c), st_) for c in row])
                    t = Table(td, colWidths=[cw]*nc)
                    t.setStyle(TableStyle([
                        ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                        ("FONTNAME",(0,0),(-1,0),FB), ("FONTSIZE",(0,0),(-1,-1),9),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LGREY]),
                        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#cccccc")),
                        ("VALIGN",(0,0),(-1,-1),"TOP"), ("TOPPADDING",(0,0),(-1,-1),4),
                        ("BOTTOMPADDING",(0,0),(-1,-1),4), ("LEFTPADDING",(0,0),(-1,-1),6),
                    ]))
                    story.append(t); story.append(Spacer(1,3*mm))
                continue

            if stripped.startswith(("- ","* ","+ ")):
                story.append(Paragraph(f"\u2022\u00a0\u00a0{inline(stripped[2:])}", sBullet))
                i += 1; continue

            if stripped.startswith("> "):
                story.append(Paragraph(f"<i>{inline(stripped[2:])}</i>", sQuote))
                i += 1; continue

            if not stripped:
                story.append(Spacer(1,2*mm)); i += 1; continue

            story.append(Paragraph(inline(stripped), sBody)); i += 1

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        return buf.getvalue()
    except Exception:
        return b""

with tab_docs:
    st.subheader("Generate documentation")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Ingest a repository first using the Ingest tab.")
    else:
        st.caption(f"{len(result['files'])} files, {len(result['chunks'])} chunks")

        doc_type = st.selectbox(
            "Document type",
            options=["api", "readme", "guide"],
            format_func=lambda x: {
                "api":    "API Reference (structured per-module docs)",
                "readme": "README (project overview for the repo)",
                "guide":  "Beginner Guide (plain-language walkthrough)",
            }[x],
        )

        doc_history = st.session_state.get("doc_history", {})
        type_labels = {"api": "API Reference", "readme": "README", "guide": "Beginner Guide"}

        existing_doc = doc_history.get(doc_type)

        doc_job_id = st.session_state.get("doc_job_id")
        if doc_job_id and "last_doc" not in st.session_state:
            try:
                job_state = poll_doc_job(doc_job_id)
            except BackendError as exc:
                st.error(f"Failed to poll: {exc}")
                job_state = None

            if job_state:
                if job_state["status"] == "running":
                    st.info("Running. Page refreshes every 5 seconds.")
                    time.sleep(5)
                    st.rerun()
                elif job_state["status"] == "failed":
                    st.error(f"Failed: {job_state.get('error', 'unknown error')}")
                    st.session_state.pop("doc_job_id", None)
                elif job_state["status"] == "done":
                    st.session_state["last_doc"] = job_state
                    dt = job_state.get("doc_type", "api")
                    st.session_state.setdefault("doc_history", {})[dt] = job_state
                    st.session_state.pop("doc_job_id", None)
                    st.rerun()

        if existing_doc:
            col1, col2, col_btn = st.columns([1, 1, 2])
            col1.metric("Files summarised", existing_doc["files_summarised"])
            col2.metric("Requests used",    existing_doc["llm_calls_used"])
            with col_btn:
                if st.button("Regenerate", type="secondary"):
                    try:
                        started = start_doc_generation(result["job_id"], doc_type)
                        st.session_state["doc_job_id"] = started["doc_job_id"]
                        st.session_state.pop("last_doc", None)
                        st.rerun()
                    except BackendError as exc:
                        st.error(f"Could not start: {exc}")
            st.divider()
            import re as _re
            clean_content = _re.sub(r"^\s*-{3,}\s*$", "", existing_doc["content"], flags=_re.MULTILINE)
            st.markdown(clean_content)

            col_md, col_pdf, col_spacer = st.columns([1, 1, 4])
            repo_name = result.get("repo_url", "repo").rstrip("/").split("/")[-1]
            doc_label = type_labels.get(doc_type, doc_type)
            safe_name = f"{repo_name}_{doc_label.replace(' ', '_')}"
            with col_md:
                st.download_button(
                    label="Download Markdown",
                    data=clean_content,
                    file_name=f"{safe_name}.md",
                    mime="text/markdown",
                )
            with col_pdf:
                pdf_bytes = _make_pdf(
                    clean_content,
                    doc_title=f"{repo_name}: {doc_label}",
                )
                if pdf_bytes:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"{safe_name}.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.caption("PDF unavailable. Install reportlab.")
        else:
            # Not generated yet for this type
            st.info(f"No {type_labels.get(doc_type, doc_type)} generated yet.")
            if st.button("Generate", type="primary"):
                try:
                    started = start_doc_generation(result["job_id"], doc_type)
                    st.session_state["doc_job_id"] = started["doc_job_id"]
                    st.session_state.pop("last_doc", None)
                    st.rerun()
                except BackendError as exc:
                    st.error(f"Could not start: {exc}")


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def _render_answer(answer: str, question: str, sources: list, key_suffix: str):
    """
    Renders answer segments. Code blocks get an Add to PR button.
    """
    segments = _re.split(r"(```(?:\w+)?\n.*?```)", answer, flags=_re.DOTALL)
    code_index = 0

    for segment in segments:
        if not segment.strip():
            continue
        code_match = _re.match(r"```(\w+)?\n(.*?)```", segment, _re.DOTALL)
        if code_match:
            lang = code_match.group(1) or "python"
            code = code_match.group(2).rstrip()
            btn_key = f"pr_btn_{key_suffix}_{code_index}"

            st.code(code, language=lang)

            if st.button("Add to PR", key=btn_key):
                st.session_state["_pr_form"] = {
                    "key_suffix": key_suffix,
                    "code_index": code_index,
                    "code":       code,
                    "question":   question,
                    "source_path": sources[0]["file_path"] if sources else "",
                }
                st.rerun()

            code_index += 1
        else:
            st.markdown(segment)

    if sources:
        with st.expander("Sources", expanded=False):
            for src in sources:
                st.markdown(
                    f"- `{src['file_path']}` "
                    f"**{src['symbol_name'] or src['chunk_type']}** "
                    f"(lines {src['start_line']}–{src['end_line']})"
                )

# ─────────────────────────────────────────────────────────────────────────
# RAG CODE ASSISTANT SECTION
# ─────────────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("RAG Code Assistant")
    st.caption("Ask questions about the codebase. Answers are grounded in the actual source code.")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Ingest a repository first using the Ingest tab.")
    else:
        col_info, col_clear = st.columns([5, 1])
        with col_info:
            st.caption(f"Active repo: `{result.get('repo_url', 'unknown')}`")
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state["chat_history"] = []
                st.session_state.pop("last_qa", None)
                st.session_state.pop("_pr_form", None)
                st.rerun()

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        question = st.chat_input("Ask something about the codebase...")

        if question:
            st.session_state.pop("last_qa", None)
            st.session_state.pop("_pr_form", None)

            with st.chat_message("user"):
                st.markdown(question)

            api_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_history"]
            ]

            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    try:
                        response = ask_question(
                            result["job_id"], question, history=api_history
                        )
                        answer  = response["answer"]
                        sources = response.get("sources", [])
                    except BackendError as exc:
                        answer  = f"Error: {exc}"
                        sources = []

                _render_answer(answer, question, sources, key_suffix="live")

            st.session_state["chat_history"].append({"role": "user",      "content": question})
            st.session_state["chat_history"].append({"role": "assistant",  "content": answer})
            st.session_state["last_qa"] = {
                "question": question,
                "answer":   answer,
                "sources":  sources,
            }

        else:
            last_qa = st.session_state.get("last_qa")
            if last_qa:
                with st.chat_message("user"):
                    st.markdown(last_qa["question"])
                with st.chat_message("assistant"):
                    _render_answer(
                        last_qa["answer"], last_qa["question"],
                        last_qa.get("sources", []), key_suffix="last"
                    )

        pr_form = st.session_state.get("_pr_form")
        if pr_form:
            st.divider()
            st.markdown("**Save this code change for a PR**")
            with st.container(border=True):
                f_title = st.text_input(
                    "Change title",
                    value=pr_form["question"][:80],
                    key="pr_form_title",
                )
                f_path = st.text_input(
                    "File path",
                    value=pr_form["source_path"],
                    placeholder="e.g. code.py",
                    key="pr_form_path",
                    help="Where to commit this file in the repo.",
                )
                f_content = st.text_area(
                    "Content",
                    value=pr_form["code"],
                    height=220,
                    key="pr_form_content",
                )
                col_s, col_c = st.columns(2)
                with col_s:
                    if st.button("Save for PR", type="primary", key="pr_form_save"):
                        st.session_state.setdefault("code_changes", []).append({
                            "id":              f"qa_{len(st.session_state['code_changes'])}",
                            "title":           f_title or pr_form["question"][:80],
                            "content":         f_content,
                            "file_path":       f_path.strip() or None,
                            "source_question": pr_form["question"],
                        })
                        st.session_state.pop("_pr_form", None)
                        st.success("Saved. Switch to the **Create PR** tab.")
                        st.rerun()
                with col_c:
                    if st.button("Cancel", key="pr_form_cancel"):
                        st.session_state.pop("_pr_form", None)
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# CREATE PR SECTION
# ─────────────────────────────────────────────────────────────────────────
with tab_pr:
    st.subheader("Create a Pull Request")
    st.caption("Commit generated documentation or code changes to a branch and open a pull request.")

    ingest_result = st.session_state.get("last_result")
    doc_result    = st.session_state.get("last_doc")
    github_token  = st.session_state.get("github_token")

    has_ingest = ingest_result is not None
    has_doc    = doc_result is not None and doc_result.get("content")
    has_github = bool(github_token)

    if ingest_result and ingest_result.get("_restored_after_oauth"):
        st.info(
            "Session restored after GitHub login. "
            "If docs were generated before login, re-generate them in the Generate Docs tab."
        )

    st.markdown("**Checklist**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Repo ingested",   "Yes" if has_ingest else "No")
    col2.metric("Docs generated",  "Yes" if has_doc    else "No")
    col3.metric("GitHub connected","Yes" if has_github else "No")

    st.divider()

    if not has_github:
        st.info("Connect your GitHub account to enable pull request creation.")
        try:
            login_url = get_github_login_url()
            if ingest_result:
                st.session_state["pre_oauth_job_id"] = ingest_result.get("job_id", "")
            st.markdown(
                f'<a href="{login_url}" target="_self" style="display:inline-block;'
                f'padding:0.5rem 1.2rem;background:#1F3864;color:white;border-radius:6px;'
                f'text-decoration:none;font-weight:600;font-size:1rem;">Connect GitHub</a>',
                unsafe_allow_html=True,
            )
            st.caption("Opens GitHub's authorisation page in this tab. You will be returned here automatically.")
        except BackendError as exc:
            st.error(f"Cannot build GitHub login URL: {exc}. Check GITHUB_CLIENT_ID in backend/.env.")
    else:
        status = get_github_status(github_token)
        if status.get("connected"):
            st.success(f"Connected as **{status['github_name']}** (@{status['github_login']})")
        else:
            st.warning("GitHub token expired or invalid. Reconnect below.")
            st.session_state.pop("github_token", None)
            try:
                login_url = get_github_login_url()
                st.markdown(
                    f'<a href="{login_url}" target="_self" style="display:inline-block;'
                    f'padding:0.4rem 1rem;background:#1F3864;color:white;border-radius:6px;'
                    f'text-decoration:none;font-weight:600;">Reconnect GitHub</a>',
                    unsafe_allow_html=True,
                )
            except BackendError:
                pass
            st.stop()

        if st.button("Disconnect GitHub", type="secondary"):
            st.session_state.pop("github_token", None)
            st.rerun()

    if st.session_state.pop("_pr_reset_pending", False):
        for k in list(st.session_state.keys()):
            if k.startswith(("doc_select_", "change_include_")):
                st.session_state.pop(k, None)
        st.session_state["branch_name_input"] = ""

    if has_ingest and has_github:
        st.divider()
        st.markdown("**Pull request details**")

        ingested_url = ingest_result.get("repo_url", "")
        auto_repo = ingested_url.split("github.com/")[-1].rstrip("/") if "github.com/" in ingested_url else ""

        repo_input = st.text_input(
            "Repository (owner/repo)",
            value=auto_repo,
            help="Auto-filled from the ingested URL. Change if needed.",
        )
        branch_input = st.text_input(
            "Branch name (optional)",
            placeholder="Leave blank to auto-generate",
            key="branch_name_input",
        )

        if "doc_history" not in st.session_state:
            st.session_state["doc_history"] = {}
        if doc_result and doc_result.get("content"):
            st.session_state["doc_history"][doc_result.get("doc_type", "api")] = doc_result

        doc_history  = st.session_state["doc_history"]
        code_changes = st.session_state.get("code_changes", [])
        type_labels  = {"api": "API Reference", "readme": "README", "guide": "Beginner Guide"}

        if not doc_history and not code_changes:
            st.warning("Nothing to commit yet. Generate docs or save a code change from the RAG Code Assistant tab.")
        else:
            selected_docs = []

            if doc_history:
                st.markdown("**Documentation**")
                for dt, info in doc_history.items():
                    char_count = len(info.get("content", ""))
                    if st.checkbox(
                        f"{type_labels.get(dt, dt.upper())} ({char_count:,} chars, "
                        f"{info.get('files_summarised', 0)} files",
                        value=True, key=f"doc_select_{dt}",
                    ):
                        selected_docs.append({"doc_type": dt, "content": info["content"], "file_path": None})

            if code_changes:
                st.markdown("**Code changes from RAG Code Assistant**")
                for i, change in enumerate(code_changes):
                    with st.expander(change["title"][:70], expanded=True):
                        col_check, col_del = st.columns([5, 1])
                        with col_check:
                            include = st.checkbox("Include", value=True, key=f"change_include_{i}")
                        with col_del:
                            if st.button("Remove", key=f"change_del_{i}"):
                                st.session_state["code_changes"].pop(i)
                                st.rerun()

                        edit_path = st.text_input(
                            "File path",
                            value=change.get("file_path") or "",
                            placeholder="e.g. app/services/rag.py",
                            key=f"change_path_{i}",
                        )
                        edit_content = st.text_area(
                            "Content",
                            value=change["content"],
                            height=250,
                            key=f"change_content_{i}",
                        )
                        if include:
                            selected_docs.append({
                                "doc_type":  "code_change",
                                "content":   edit_content,
                                "file_path": edit_path.strip() or f"docs/code_change_{i}.md",
                            })

            if selected_docs:
                file_preview = ", ".join(
                    d.get("file_path") or {
                        "api": "docs/API_Reference.md",
                        "readme": "docs/README.md",
                        "guide": "docs/Beginner_Guide.md",
                    }.get(d["doc_type"], f"docs/{d['doc_type']}.md")
                    for d in selected_docs
                )
                st.caption(f"Will commit: `{file_preview}`")

            confirmed = st.checkbox("I confirm I want to commit these files and open a pull request.")
            n = len(selected_docs)
            if st.button(
                f"Create Pull Request ({n} file{'s' if n != 1 else ''})",
                type="primary",
                disabled=not confirmed or not repo_input.strip() or not selected_docs,
            ):
                with st.spinner("Creating branch, committing files, opening pull request..."):
                    try:
                        pr_result = create_pull_request(
                            github_token=github_token,
                            job_id=ingest_result["job_id"],
                            repo_full_name=repo_input.strip(),
                            docs=selected_docs,
                            custom_branch_name=branch_input.strip() or None,
                        )
                        st.session_state["last_pr"] = pr_result
                        st.session_state["_pr_reset_pending"] = True
                        st.rerun()
                    except BackendError as exc:
                        st.error(f"PR creation failed: {exc}")

    elif has_github and not has_ingest:
        st.info("Go to the Ingest tab and ingest a repository first.")

    pr = st.session_state.get("last_pr")
    if pr:
        st.divider()
        st.success("Pull request created.")

        if pr.get("forked"):
            st.info(f"Forked to **{pr['fork_name']}** automatically. The pull request is within your fork.")

        col1, col2 = st.columns(2)
        col1.metric("PR number", f"#{pr['pr_number']}")
        col2.metric("Branch",    pr["branch_name"])

        files = pr.get("files_committed", [])
        if files:
            st.markdown("**Files committed:**")
            for f in files:
                st.markdown(f"- `{f}`")

        st.markdown(f"**Title:** {pr['pr_title']}")
        st.link_button("Open pull request on GitHub", url=pr["pr_url"], type="primary")
        if st.button("Create another"):
            st.session_state.pop("last_pr", None)
            st.session_state.pop("doc_history", None)
            st.session_state.pop("last_doc", None)
            st.session_state.pop("code_changes", None)
            for k in list(st.session_state.keys()):
                if k.startswith(("show_pr_form_", "s_title_", "s_path_",
                                  "s_content_", "s_save_", "s_cancel_",
                                  "pr_btn_", "change_path_", "change_content_",
                                  "pr_form_")):
                    st.session_state.pop(k, None)
            st.session_state.pop("_pr_form", None)
            st.session_state["_pr_reset_pending"] = True
            st.rerun()
