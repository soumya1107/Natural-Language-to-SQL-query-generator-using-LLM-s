"""
app.py
──────
NL2SQL Streamlit Application

Run with:
    streamlit run app.py
"""

import sys
import os
import time
from dataclasses import dataclass, field

import streamlit as st
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import get_db_config, get_app_config
from db.connector import DBConnector
from db.schema_fetcher import SchemaFetcher
from llm.factory import get_llm, PROVIDER_DISPLAY, PROVIDER_MODELS
from core.query_pipeline import QueryPipeline
from core.result_formatter import QueryResult, suggest_chart_columns


# ══════════════════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NL2SQL · Query in Plain English",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2a4a 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .hero h1 { color: #f8fafc; font-size: 2rem; font-weight: 700; margin: 0; }
    .hero p  { color: #94a3b8; font-size: 1rem; margin: 0.5rem 0 0; }

    /* SQL code blocks */
    .sql-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #7dd3fc;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    /* Summary card */
    .summary-card {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 0.9rem 1.25rem;
        color: #14532d;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Error card */
    .error-card {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 0.9rem 1.25rem;
        color: #7f1d1d;
        font-size: 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* History item */
    .hist-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        cursor: pointer;
        color: #334155;
    }
    .hist-item:hover { background: #e0f2fe; border-color: #38bdf8; }

    /* Metric badges */
    .metric-row { display: flex; gap: 1rem; margin: 0.75rem 0; }
    .metric-badge {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.8rem;
        color: #475569;
        font-weight: 500;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label { color: #94a3b8 !important; font-size: 0.8rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 6px 6px 0 0;
        padding: 6px 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #0f172a !important; color: white !important; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "history": [],          # list of QueryResult + question dicts
        "pipeline": None,
        "connector": None,
        "db_connected": False,
        "schema": None,
        "last_result": None,
        "last_question": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — DB connection + LLM settings
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🗄️ Database")

    db_cfg = get_db_config()

    db_host = st.text_input("Host", value=db_cfg.host, key="db_host")
    col_port, col_db = st.columns(2)
    db_port = col_port.number_input("Port", value=db_cfg.port, step=1, key="db_port")
    db_name = col_db.text_input("Database", value=db_cfg.database, key="db_name")
    db_user = st.text_input("User", value=db_cfg.user, key="db_user")
    db_pass = st.text_input("Password", value=db_cfg.password, type="password", key="db_pass")

    connect_btn = st.button("Connect", use_container_width=True, type="primary")

    if connect_btn:
        with st.spinner("Connecting…"):
            conn = DBConnector(
                host=db_host, port=int(db_port),
                user=db_user, password=db_pass, database=db_name
            )
            ok, msg = conn.test_connection()
            if ok:
                st.session_state.connector = conn
                st.session_state.db_connected = True
                # Pre-fetch schema
                fetcher = SchemaFetcher(conn)
                try:
                    st.session_state.schema = fetcher.fetch()
                    st.success(f"✓ Connected — {len(st.session_state.schema.tables)} tables")
                except Exception as e:
                    st.warning(f"Connected but schema fetch failed: {e}")
            else:
                st.session_state.db_connected = False
                st.error(f"Connection failed: {msg}")

    st.markdown("---")
    st.markdown("### 🤖 LLM Settings")

    provider_labels = list(PROVIDER_DISPLAY.values())
    provider_keys   = list(PROVIDER_DISPLAY.keys())

    selected_label = st.selectbox("Provider", provider_labels, index=0)
    selected_provider = provider_keys[provider_labels.index(selected_label)]

    # Dynamic model list
    if selected_provider == "ollama":
        # Try to list local models
        try:
            from llm.ollama_llm import OllamaLLM
            from config.settings import get_llm_config
            llm_cfg = get_llm_config()
            tmp = OllamaLLM(base_url=llm_cfg.ollama_base_url)
            local_models = tmp.list_models() or ["llama3", "mistral", "codellama"]
        except Exception:
            local_models = ["llama3", "mistral", "codellama"]
        model_list = local_models
    else:
        model_list = PROVIDER_MODELS[selected_provider]

    selected_model = st.selectbox("Model", model_list, index=0)

    # API key override (shows only for cloud providers)
    api_key_override = ""
    if selected_provider != "ollama":
        api_key_override = st.text_input(
            "API Key (leave blank to use .env)",
            type="password",
            key="api_key_override",
        )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    summarise = st.toggle("Generate plain-English summary", value=True)

    st.markdown("---")
    st.markdown("### ⚙️ Query Options")
    read_only = st.toggle("Read-only guard (block mutations)", value=True)
    max_rows = st.number_input("Max result rows", value=200, min_value=1, max_value=5000)

    if st.button("🔄 Refresh Schema", use_container_width=True):
        if st.session_state.pipeline:
            st.session_state.pipeline.refresh_schema()
            st.session_state.schema = st.session_state.pipeline.schema
            st.success("Schema refreshed.")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.success("History cleared.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero">
        <h1>🔍 NL2SQL</h1>
        <p>Ask your database anything in plain English. AI converts it to SQL and returns readable results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_query, tab_schema, tab_history = st.tabs(["💬 Query", "🗂️ Schema Explorer", "📜 History"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Query
# ══════════════════════════════════════════════════════════════════════════════

with tab_query:

    # Example prompts
    st.markdown("**Try an example:**")
    examples = [
        "Show me the top 10 customers by total order value",
        "How many orders were placed each month this year?",
        "Which products have never been ordered?",
        "List all users who signed up in the last 30 days",
        "What is the average order value by country?",
    ]
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex[:35] + "…" if len(ex) > 35 else ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["prefill_question"] = ex

    # Question input
    prefill = st.session_state.pop("prefill_question", st.session_state.last_question)
    question = st.text_area(
        "Your question",
        value=prefill,
        height=90,
        placeholder="e.g. Show me all orders from customers in Germany last quarter",
        label_visibility="collapsed",
    )

    col_run, col_gap = st.columns([1, 4])
    run_btn = col_run.button("▶  Run Query", type="primary", use_container_width=True)

    if run_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not st.session_state.db_connected:
            st.error("Connect to a database first (use the sidebar).")
        else:
            # Build LLM
            try:
                llm = get_llm(
                    provider=selected_provider,
                    model=selected_model,
                    temperature=temperature,
                    api_key=api_key_override if api_key_override else None,
                )
            except Exception as e:
                st.error(f"LLM init error: {e}")
                st.stop()

            # Override app config settings from sidebar
            from config.settings import get_app_config
            app_cfg = get_app_config()
            app_cfg.read_only_guard = read_only
            app_cfg.max_result_rows = int(max_rows)

            # Build / reuse pipeline
            if (
                st.session_state.pipeline is None
                or st.session_state.pipeline.llm.model != llm.model
                or st.session_state.pipeline.llm.provider != llm.provider
            ):
                st.session_state.pipeline = QueryPipeline(
                    connector=st.session_state.connector,
                    llm=llm,
                    summarise=summarise,
                )
            else:
                st.session_state.pipeline.llm = llm
                st.session_state.pipeline.summarise = summarise

            with st.spinner(f"Thinking with {PROVIDER_DISPLAY[selected_provider]} / {selected_model}…"):
                result = st.session_state.pipeline.run(question)
                st.session_state.last_result = result
                st.session_state.last_question = question

                # Save to history
                app_cfg_h = get_app_config()
                history_entry = {
                    "question": question,
                    "result": result,
                    "provider": PROVIDER_DISPLAY[selected_provider],
                    "model": selected_model,
                }
                st.session_state.history.insert(0, history_entry)
                if len(st.session_state.history) > app_cfg_h.history_max:
                    st.session_state.history = st.session_state.history[:app_cfg_h.history_max]

    # ── Display results ────────────────────────────────────────────────────
    result: QueryResult | None = st.session_state.last_result
    if result:

        if result.error:
            st.markdown(
                f'<div class="error-card">⚠️ {result.error}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Metrics row
            st.markdown(
                f"""
                <div class="metric-row">
                    <span class="metric-badge">📊 {result.row_count} rows</span>
                    <span class="metric-badge">⏱ {result.execution_time_ms:.1f} ms</span>
                    <span class="metric-badge">🤖 {PROVIDER_DISPLAY.get(selected_provider, selected_provider)} / {selected_model}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # SQL
            st.markdown("**Generated SQL**")
            st.markdown(
                f'<div class="sql-box">{result.sql}</div>',
                unsafe_allow_html=True,
            )

            # Copy SQL button
            st.code(result.sql, language="sql")   # fallback for copying

            # Summary
            if result.summary:
                st.markdown(
                    f'<div class="summary-card">📝 {result.summary}</div>',
                    unsafe_allow_html=True,
                )

            if result.is_empty:
                st.info("Query returned no rows.")
            else:
                st.markdown("**Results**")
                df = result.to_dataframe()
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download CSV",
                    data=csv,
                    file_name="nl2sql_result.csv",
                    mime="text/csv",
                )

                # Auto chart suggestion
                chart_cols = suggest_chart_columns(df)
                if chart_cols["x"] and chart_cols["y"]:
                    with st.expander("📈 Auto Chart", expanded=False):
                        try:
                            chart_df = df[[chart_cols["x"], chart_cols["y"]]].copy()
                            chart_df[chart_cols["y"]] = pd.to_numeric(chart_df[chart_cols["y"]], errors="coerce")
                            chart_df = chart_df.dropna().sort_values(chart_cols["y"], ascending=False).head(20)
                            st.bar_chart(chart_df.set_index(chart_cols["x"]))
                        except Exception as e:
                            st.info(f"Chart unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Schema Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab_schema:
    if not st.session_state.db_connected:
        st.info("Connect to a database in the sidebar to explore its schema.")
    elif st.session_state.schema is None:
        st.info("Schema not yet loaded.")
    else:
        schema = st.session_state.schema
        st.markdown(f"**{schema.database}** — {len(schema.tables)} tables")

        search = st.text_input("🔍 Filter tables / columns", placeholder="Search…", label_visibility="collapsed")

        for table_name, table in schema.tables.items():
            if search and search.lower() not in table_name.lower() and not any(
                search.lower() in col.name.lower() for col in table.columns
            ):
                continue

            with st.expander(f"📋 {table_name}  ({table.row_count:,} rows)"):
                col_data = []
                for col in table.columns:
                    tags = []
                    if col.key == "PRI": tags.append("🔑 PK")
                    if col.key == "UNI": tags.append("✦ UNIQUE")
                    if col.key == "MUL": tags.append("⊕ INDEX")
                    if not col.nullable: tags.append("NOT NULL")
                    if col.extra: tags.append(col.extra.upper())
                    col_data.append({
                        "Column": col.name,
                        "Type": col.data_type,
                        "Flags": "  ".join(tags),
                    })
                st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

                if table.foreign_keys:
                    st.markdown("**Foreign Keys**")
                    for fk in table.foreign_keys:
                        st.markdown(f"↳ `{fk['column']}` → `{fk['ref_table']}.{fk['ref_column']}`")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    if not st.session_state.history:
        st.info("No queries yet. Run your first query above!")
    else:
        st.markdown(f"**{len(st.session_state.history)} queries** in this session")

        for i, entry in enumerate(st.session_state.history):
            q   = entry["question"]
            r   = entry["result"]
            prv = entry["provider"]
            mdl = entry["model"]

            status = "✅" if not r.error else "❌"
            with st.expander(f"{status}  {q[:80]}{'…' if len(q) > 80 else ''}  ·  {prv}/{mdl}"):
                if r.error:
                    st.markdown(
                        f'<div class="error-card">{r.error}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{r.row_count} rows** · {r.execution_time_ms:.1f} ms")
                    st.code(r.sql, language="sql")
                    if r.summary:
                        st.markdown(
                            f'<div class="summary-card">📝 {r.summary}</div>',
                            unsafe_allow_html=True,
                        )
                    if not r.is_empty:
                        st.dataframe(r.to_dataframe(), use_container_width=True, hide_index=True)

                if st.button("Re-run this question", key=f"rerun_{i}"):
                    st.session_state["prefill_question"] = q
                    st.rerun()
