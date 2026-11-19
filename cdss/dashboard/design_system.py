"""Visual primitives for the Streamlit ML workbench."""

from __future__ import annotations

import streamlit as st


def inject_design_system() -> None:
    """Apply a restrained editorial visual system to the Streamlit shell."""
    st.markdown(
        """
        <style>
            :root {
                --ink: #11231f;
                --muted: #5f706c;
                --paper: #f4f6f1;
                --surface: #fbfcf9;
                --line: #d4ddd6;
                --signal: #0b6a55;
                --signal-dark: #064435;
                --signal-soft: #dceee7;
                --amber: #b75b12;
                --alert: #a9382d;
                --alert-soft: #f7e8e5;
            }
            .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
            [data-testid="stHeader"] { background: rgba(244, 246, 241, .9); border-bottom: 1px solid rgba(17, 35, 31, .08); }
            .block-container { max-width: 1450px; padding-top: 2.2rem; padding-bottom: 4rem; }
            h1, h2, h3 { color: var(--ink) !important; font-family: Georgia, "Times New Roman", serif; font-weight: 500 !important; letter-spacing: -.025em; }
            h2 { font-size: 1.65rem !important; }
            h3 { font-size: 1.15rem !important; }
            p, label, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
            [data-testid="stSidebar"] { background: #102822; border-right: 1px solid #234239; }
            [data-testid="stSidebar"] * { color: #ecf4ef !important; }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #b7cec4 !important; }
            .sidebar-label, .eyebrow { color: var(--signal) !important; font-family: Inter, ui-sans-serif, sans-serif !important; font-size: .68rem !important; font-weight: 800 !important; letter-spacing: .16em !important; text-transform: uppercase; }
            .sidebar-label { color: #9bcab8 !important; }
            .masthead { position: relative; overflow: hidden; margin: 0 0 1.25rem; padding: 2rem 2.15rem 2.15rem; color: #f5fbf7; background: linear-gradient(118deg, #0b2f28 0%, #0c5948 56%, #0a7861 100%); border-bottom: 5px solid #c99845; box-shadow: 0 18px 45px rgba(10, 52, 43, .16); }
            .masthead::after { content: ""; position: absolute; width: 30rem; height: 30rem; right: -12rem; top: -19rem; border: 1px solid rgba(255,255,255,.18); border-radius: 50%; box-shadow: 0 0 0 2.8rem rgba(255,255,255,.04), 0 0 0 7rem rgba(255,255,255,.035); }
            .masthead-grid { position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 2rem; align-items: end; }
            .masthead-kicker { color: #a8d8c5; font-family: Inter, ui-sans-serif, sans-serif; font-size: .7rem; font-weight: 800; letter-spacing: .17em; text-transform: uppercase; }
            .masthead h1 { margin: .35rem 0 .55rem !important; color: #fff !important; font-size: clamp(2.2rem,4.2vw,4rem) !important; line-height: .98 !important; }
            .masthead p { max-width: 46rem; margin: 0; color: #d6e7df; font-size: 1rem; line-height: 1.6; }
            .masthead-stat { min-width: 12rem; padding: .9rem 0 .2rem 1.3rem; border-left: 1px solid rgba(255,255,255,.4); }
            .masthead-stat span { display: block; color: #b9d8ca; font-family: Inter, ui-sans-serif, sans-serif; font-size: .68rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
            .masthead-stat strong { display: block; margin-top: .25rem; color: #fff; font-family: Georgia, "Times New Roman", serif; font-size: 1.18rem; font-weight: 500; }
            .workflow-heading { display: flex; gap: .9rem; align-items: baseline; margin: 1.6rem 0 .8rem; }
            .workflow-number { color: var(--amber); font-family: Georgia, "Times New Roman", serif; font-size: 1.7rem; line-height: 1; }
            .workflow-heading h2 { margin: 0 !important; }
            .workflow-heading p { margin: .35rem 0 0; color: var(--muted); font-size: .9rem; }
            .workbench-panel { height: 100%; padding: 1.15rem 1.2rem 1.25rem; background: var(--surface); border-top: 3px solid var(--signal); box-shadow: 0 10px 24px rgba(28,57,48,.06); }
            .workbench-panel--warm { border-top-color: var(--amber); }
            .panel-title { margin: 0 0 .22rem; color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-size: 1.25rem; }
            .panel-copy { margin: 0; color: var(--muted); font-size: .84rem; line-height: 1.5; }
            [data-testid="stMetric"] { padding: .8rem .15rem .7rem; background: transparent; border-top: 1px solid var(--line); }
            [data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
            [data-testid="stMetricValue"] { color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-size: 1.55rem; font-weight: 500; }
            [data-testid="stTabs"] [role="tablist"] { gap: 1.4rem; border-bottom: 1px solid var(--line); }
            [data-testid="stTabs"] [role="tab"] { height: 3rem; padding: 0 .1rem; color: var(--muted); font-size: .72rem; font-weight: 800; letter-spacing: .11em; }
            [data-testid="stTabs"] [aria-selected="true"] { color: var(--signal) !important; }
            [data-testid="stTabs"] [role="tab"] p { font-size: inherit; }
            [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--line); border-radius: 0; }
            [data-testid="stExpander"] summary { color: var(--ink); font-size: .82rem; font-weight: 750; }
            [data-testid="stFileUploaderDropzone"] { background: #f1f6f2; border: 1px dashed #7ba997; border-radius: 0; }
            [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div { background: #fff; border-radius: 0; }
            .stButton > button, [data-testid="stFormSubmitButton"] > button { border-radius: 0 !important; background: var(--signal) !important; border: 1px solid var(--signal) !important; color: #fff !important; font-weight: 750 !important; letter-spacing: .02em; }
            .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background: var(--signal-dark) !important; border-color: var(--signal-dark) !important; }
            [data-testid="stDataFrame"] { border: 1px solid var(--line); }
            .result-band { margin-top: 1.15rem; padding: 1.1rem 1.25rem; border-left: 5px solid var(--signal); background: var(--signal-soft); }
            .result-band--alert { border-left-color: var(--alert); background: var(--alert-soft); }
            .result-kicker { margin: 0; color: var(--muted); font-family: Inter, ui-sans-serif, sans-serif; font-size: .68rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
            .result-title { margin: .22rem 0 0; color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-size: 1.75rem; font-weight: 500; }
            .data-note { margin: .8rem 0 1.2rem; padding: .8rem 1rem; border-left: 2px solid #c99845; color: var(--muted); font-size: .86rem; line-height: 1.55; }
            @media (max-width: 720px) { .masthead { padding: 1.5rem; } .masthead-grid { grid-template-columns: 1fr; gap: 1rem; } .masthead-stat { border-left: 0; border-top: 1px solid rgba(255,255,255,.35); padding: .8rem 0 0; } .block-container { padding-top: 1.1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_masthead(selected_model: str, record_count: int) -> None:
    """Render the dashboard's decision-focused visual anchor."""
    st.markdown(
        f"""
        <section class="masthead">
          <div class="masthead-grid">
            <div>
              <div class="masthead-kicker">Signal Lab / Educational ML workbench</div>
              <h1>Breast tumour<br>classification</h1>
              <p>Compare five WDBC pipelines, inspect their evidence, and run one reviewed prediction through a chosen classifier.</p>
            </div>
            <div class="masthead-stat">
              <span>Current default</span><strong>{selected_model}</strong>
              <span style="margin-top:.65rem">Reference cohort</span><strong>{record_count} WDBC records</strong>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_heading(number: str, title: str, description: str) -> None:
    """Render a reusable numbered workbench heading."""
    st.markdown(
        f"""
        <div class="workflow-heading">
          <span class="workflow-number">{number}</span>
          <div><h2>{title}</h2><p>{description}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, description: str, eyebrow: str, warm: bool = False) -> None:
    """Render a short contextual panel before a Streamlit control group."""
    warm_class = " workbench-panel--warm" if warm else ""
    st.markdown(
        f"""
        <section class="workbench-panel{warm_class}">
          <p class="eyebrow">{eyebrow}</p>
          <p class="panel-title">{title}</p>
          <p class="panel-copy">{description}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_result_band(diagnosis: str) -> None:
    """Render a clear diagnostic-status treatment after reviewed inference."""
    is_malignant = diagnosis == "Malignant"
    tone_class = " result-band--alert" if is_malignant else ""
    st.markdown(
        f"""
        <section class="result-band{tone_class}">
          <p class="result-kicker">Reviewed educational model output</p>
          <p class="result-title">{diagnosis}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
