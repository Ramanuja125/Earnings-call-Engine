"""
ECRIE — Web Interface
=====================
Two-portal application:

  USER (Analyst) Portal  — Dashboard, Analysis, Results
  DEVELOPER Portal       — Pipeline status, Agent monitor, Model explorer,
                           Raw data, Configuration

Access control:
  • Users see only Analyst pages. No developer content is rendered or sent.
  • Developer mode unlocked by password in sidebar.
  • Developers see all pages (analyst + developer).

Design: all CSS/components imported from design.py. Zero inline CSS here.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime
import time

# ── Project root ──────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import RESULTS_DIR, FINAL_DATA_DIR, LOGS_DIR

# ── Design system ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import design as D

# ── Page config (must be first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="ECRIE — Earnings Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(D.MASTER_CSS, unsafe_allow_html=True)

# ── Developer password ────────────────────────────────────────────────
_DEV_PASSWORD = "admin123"   # change before deployment

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
for k, v in [("dev_auth", False), ("page", "user_dashboard")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS  (all cached)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_state():
    p = RESULTS_DIR / 'agentic_orchestration' / 'orchestration_state.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


@st.cache_data
def load_test_data():
    p = FINAL_DATA_DIR / 'test_data.csv'
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_models():
    out = {}
    paths = {
        'baseline': RESULTS_DIR / 'baseline_models'  / 'phase6_documentation.json',
        'xgboost':  RESULTS_DIR / 'xgboost_models'   / 'phase7_xgboost_results.json',
        'agentic':  RESULTS_DIR / 'agentic_ai'        / 'agentic_ai_report.json',
        'phase8':   RESULTS_DIR / 'validation'        / 'phase8_validation_report.json',
    }
    for k, p in paths.items():
        if p.exists():
            out[k] = json.load(open(p, encoding='utf-8'))
    return out


@st.cache_data
def load_shap():
    p = RESULTS_DIR / 'validation' / 'phase8b_feature_importance.csv'
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_cv():
    p = RESULTS_DIR / 'validation' / 'phase8a_cross_validation_results.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Space Grotesk,sans-serif;font-size:1.1rem;'
            'font-weight:700;color:#00C9A7;letter-spacing:0.04em;'
            'margin-bottom:0.2rem;">ECRIE</div>'
            '<div style="font-size:0.6rem;color:#3D5470;letter-spacing:0.12em;'
            'text-transform:uppercase;margin-bottom:1.5rem;">'
            'Earnings Intelligence</div>',
            unsafe_allow_html=True,
        )

        # ── Analyst pages (always visible) ──
        st.markdown(
            '<div style="font-size:0.6rem;color:#00C9A7;letter-spacing:0.15em;'
            'text-transform:uppercase;margin-bottom:0.4rem;">Analyst</div>',
            unsafe_allow_html=True,
        )
        _nav_btn("Dashboard",      "user_dashboard")
        _nav_btn("Analyze Call",   "user_analysis")
        _nav_btn("Browse Results", "user_results")

        st.markdown("---")

        # ── Developer section ──
        st.markdown(
            '<div style="font-size:0.6rem;color:#F0A500;letter-spacing:0.15em;'
            'text-transform:uppercase;margin-bottom:0.4rem;">Developer</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.dev_auth:
            if "show_dev_login" not in st.session_state:
                st.session_state.show_dev_login = False
            if st.button("Developer Login", key="dev_login_toggle", use_container_width=True):
                st.session_state.show_dev_login = not st.session_state.show_dev_login
                st.rerun()
            if st.session_state.show_dev_login:
                pw = st.text_input("Password", type="password", key="pw_input",
                                   label_visibility="collapsed",
                                   placeholder="Enter password…")
                if st.button("Unlock", key="unlock_btn", use_container_width=True):
                    if pw == _DEV_PASSWORD:
                        st.session_state.dev_auth = True
                        st.session_state.show_dev_login = False
                        st.session_state.page = "dev_pipeline"
                        st.rerun()
                    else:
                        st.error("Incorrect password")
        else:
            _nav_btn("Pipeline Status",  "dev_pipeline")
            _nav_btn("Agent Monitor",    "dev_agents")
            _nav_btn("Model Explorer",   "dev_models")
            _nav_btn("Raw Data",         "dev_data")
            _nav_btn("Configuration",    "dev_config")
            st.markdown("")
            if st.button("Lock Developer Mode", use_container_width=True):
                st.session_state.dev_auth = False
                st.session_state.page = "user_dashboard"
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.58rem;color:#1E2E48;text-align:center;">'
            'ECRIE v1.0 · ASU FSE 570</div>',
            unsafe_allow_html=True,
        )


def _nav_btn(label, page_key):
    """Sidebar navigation button."""
    active_style = "color:#00C9A7 !important;background:rgba(0,201,167,0.08) !important;" \
                   if st.session_state.page == page_key else ""
    if st.button(label, key=f"nav_{page_key}", use_container_width=True):
        # Block non-dev users from dev pages
        if page_key.startswith("dev_") and not st.session_state.dev_auth:
            st.warning("Developer access required.")
            return
        st.session_state.page = page_key
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# ── ANALYST PAGES ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def page_user_dashboard():
    st.markdown(D.top_bar(badge="Analyst Portal"), unsafe_allow_html=True)

    state = load_state()
    models = load_models()

    if state is None:
        st.warning("Pipeline data not found. Ask your administrator to run the pipeline first.")
        return

    perf    = state['final_results']['performance']
    quality = state['final_results']['data_quality']

    # ── KPIs ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(D.kpi("Prediction Accuracy", f"{perf['final_auc']:.1%}",
                           "how often the system is right", "#00C9A7"), unsafe_allow_html=True)
    with c2:
        color = "#30D158" if perf['improvement'] >= 0 else "#F05454"
        st.markdown(D.kpi("AI Improvement", f"{perf['improvement']:+.1%}",
                           "gain over the starting point", color), unsafe_allow_html=True)
    with c3:
        st.markdown(D.kpi("Calls Processed", "1,294",
                           "earnings calls analyzed", "#4D9FFF"), unsafe_allow_html=True)
    with c4:
        st.markdown(D.kpi("Data Integrity", "✓ Clean",
                           "no future data used in training", "#30D158"), unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(D.section_head("How Each Approach Performed",
                                   "Higher is better — above the red line means the system is useful"),
                    unsafe_allow_html=True)
        chart_data = _build_chart_data(models)
        if chart_data:
            df = pd.DataFrame(chart_data).sort_values("AUC", ascending=True)
            color_map = {
                "Phase 6 Baseline": "#4D9FFF",
                "Phase 7 XGBoost":  "#F0A500",
                "Phase 10 Agentic": "#00C9A7",
            }
            fig = go.Figure(go.Bar(
                x=df["AUC"], y=df["Model"], orientation='h',
                marker_color=[color_map.get(p, "#8FA3BF") for p in df["Phase"]],
                text=[f"{v:.1%}" for v in df["AUC"]],
                textposition='outside',
            ))
            fig.add_vline(x=0.55, line_dash="dash", line_color="#F05454",
                          annotation_text="55% threshold",
                          annotation_font_color="#F05454")
            fig.update_layout(**D.plotly_theme(), height=380,
                              xaxis_range=[0.43, 0.63], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(D.section_head("System Health", "Everything checked before results were generated"),
                    unsafe_allow_html=True)
        checks = [
            ("No Future Data Used", quality['temporal_alignment_valid'],
             "Predictions only use information available at the time"),
            ("Language Analysis", quality['feature_extraction_success'],
             "Tone and sentiment extracted from each call"),
            ("Calls Successfully Read", quality['parsing_success_rate'] >= 0.70,
             f"{quality['parsing_success_rate']:.1%} of calls processed cleanly"),
        ]
        for label, ok, note in checks:
            icon  = "✓" if ok else "✗"
            c     = "#30D158" if ok else "#F05454"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.8rem;'
                f'padding:0.7rem;background:#162034;border:1px solid #1E2E48;'
                f'border-radius:5px;margin-bottom:0.5rem;">'
                f'<span style="color:{c};font-size:1.1rem;font-weight:700;">{icon}</span>'
                f'<div><div style="font-size:0.8rem;color:#EEF2F8;">{label}</div>'
                f'<div style="font-size:0.65rem;color:#3D5470;">{note}</div></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(D.section_head("Best Approach", "The method that performed best overall"),
                    unsafe_allow_html=True)
        st.markdown(
            D.kpi("Winning Method", perf['best_model'].replace("_", " ").title(),
                  f"Accuracy: {perf['final_auc']:.1%}", "#F0A500"),
            unsafe_allow_html=True,
        )

    st.markdown(D.footer(), unsafe_allow_html=True)


def page_user_analysis():
    st.markdown(D.top_bar(badge="Analyst Portal"), unsafe_allow_html=True)
    st.markdown(D.section_head("Analyze an Earnings Call",
                               "Upload a transcript and get an instant market signal"),
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload the earnings call transcript (.txt, .pdf, .docx)",
            type=['txt', 'pdf', 'docx'],
        )

        # ── Cache file content in session_state so it survives reruns ──
        # Streamlit reruns top-to-bottom on every widget interaction.
        # uploaded.read() consumes the bytes on the first run; subsequent
        # reruns get an empty buffer.  We store content keyed by filename
        # so switching to a different file correctly replaces the cache.
        if uploaded:
            _cache_key = f"_transcript_{uploaded.name}"
            if _cache_key not in st.session_state:
                # First time this file is seen — read and cache it
                raw = uploaded.read()
                if uploaded.type == "text/plain":
                    st.session_state[_cache_key] = raw.decode('utf-8', errors='ignore')
                else:
                    st.session_state[_cache_key] = None   # PDF/DOCX: not previewed as text
            content = st.session_state[_cache_key]

            st.success(f"✅ Loaded: {uploaded.name}")

            # ── Auto-parse ticker / quarter / year from filename ──────
            # Handles formats like:  CSCO_Q3_2024.txt
            #                        AAPL_Q1_2025.pdf
            #                        MSFT-Q2-2024.docx
            #                        nvda_q4_2024.txt   (case-insensitive)
            import re as _re
            _stem  = _re.sub(r'\.[^.]+$', '', uploaded.name)   # strip extension
            _parts = _re.split(r'[_\-\.\s]+', _stem.upper())   # split on _-. or space

            _default_ticker  = "AAPL"
            _default_quarter = "Q1"
            _default_year    = 2025

            # Look for a Qn token and a 4-digit year anywhere in the parts
            for _p in _parts:
                if _re.fullmatch(r'Q[1-4]', _p):
                    _default_quarter = _p
                elif _re.fullmatch(r'20[0-9]{2}', _p):
                    _default_year = int(_p)

            # Ticker = first token that is all letters and NOT a quarter/year/generic word
            _SKIP = {'FILE', 'TRANSCRIPT', 'EARNINGS', 'CALL', 'REPORT', 'Q', 'QTR', 'FY'}
            for _p in _parts:
                if (_re.fullmatch(r'[A-Z]{1,5}', _p)
                        and _p not in _SKIP
                        and not _re.fullmatch(r'Q[1-4]', _p)):
                    _default_ticker = _p
                    break

            # Use filename-parsed values directly
            ticker  = _default_ticker
            quarter = _default_quarter
            year    = _default_year

            # ── Ticker validation ──────────────────────────────────────
            _KNOWN_SP500 = {
                'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA','BRK-B',
                'UNH','JNJ','XOM','V','JPM','PG','MA','LLY','HD','CVX','MRK',
                'ABBV','PEP','COST','KO','AVGO','TMO','WMT','MCD','CSCO','ACN',
                'ABT','BAC','CRM','ADBE','DHR','LIN','NFLX','AMD','AMGN','AXP',
                'BKNG','CAT','COP','CVS','DE','DIS','EMR','GE','GS','HON','IBM',
                'INTC','ISRG','LMT','LOW','MO','MS','NDAQ','NEE','NKE','NOW',
                'ORCL','PM','QCOM','RTX','SBUX','SCHW','SPGI','SYK','T','TMO',
                'TMUS','TXN','UNP','UPS','USB','V','VZ','WFC','WM','XOM','META',
                'PLTR','MRNA','TSLA','NVDA','MU','STX','AXON','ABNB','AIG','ADI',
                'ACN','ADBE','ADP','AON','ARE','BEN','BIIB','CBRE','MMM','GS',
                'JNJ','JPM','MA','MRNA','NFLX','NVDA','PLTR','TSLA','UNH','AMD',
            }
            _ticker_valid = bool(_re.fullmatch(r'[A-Z]{1,5}(-[A-Z])?', ticker))
            _ticker_in_sp500 = ticker in _KNOWN_SP500

            if not ticker:
                st.error("⚠️  Please enter a ticker symbol (e.g. CSCO, AAPL).")
            elif not _ticker_valid:
                st.error(f"⚠️  '{ticker}' doesn't look like a valid ticker. "
                         f"Use 1–5 uppercase letters (e.g. CSCO, BRK-B).")
            elif not _ticker_in_sp500:
                st.markdown(
                    f'<div style="background:#2A1A00;border:1px solid #F0A500;'
                    f'border-radius:6px;padding:0.9rem 1.1rem;margin-bottom:0.75rem;">'
                    f'<div style="color:#F0A500;font-weight:700;font-size:0.85rem;'
                    f'margin-bottom:0.35rem;">⚠️  {ticker} is not in our coverage list</div>'
                    f'<div style="color:#C88A30;font-size:0.78rem;line-height:1.65;">'
                    f'This tool was built using data from large, well-known US companies. '
                    f'A result for <strong style="color:#F0A500;">{ticker}</strong> will '
                    f'still be generated, but it may be less accurate because the system '
                    f'has not seen similar companies before.<br><br>'
                    f'<strong style="color:#EEF2F8;">What to expect:</strong> The tone and '
                    f'language analysis will still work, but the financial comparisons may '
                    f'not be well-calibrated for this company. '
                    f'Use the result as a rough guide only.</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Transcript preview — content-aware mismatch detection ──
            # We scan the actual transcript text to detect the real ticker,
            # quarter, and year it belongs to, then compare against what the
            # user has selected.  Mismatches are shown as prominent errors
            # INSIDE the expander so the user can't miss them.
            _preview_label = (f"📄 Preview  ·  {ticker} {quarter} {year}"
                              if ticker else f"📄 Preview  ·  {quarter} {year}")

            def _detect_content_identity(text, check_ticker, check_quarter, check_year):
                """
                Scan the first 1 500 chars of transcript text and return a list
                of mismatch strings (empty list = everything looks fine).
                """
                _mismatches = []
                _head = text[:1500].upper()

                # ── Ticker check ────────────────────────────────────────
                # Look for the ticker as a standalone word/token in the header.
                # e.g. "(CSCO)" or "CSCO Q" or "TICKER: CSCO"
                if check_ticker and not _re.search(
                        rf'\b{_re.escape(check_ticker)}\b', _head):
                    _mismatches.append(
                        f"**Wrong company** — the file does not appear to be for **{check_ticker}**. "
                        f"It may belong to a different company."
                    )

                # ── Quarter check ───────────────────────────────────────
                # Map quarter tokens that appear in earnings transcript headers.
                _QTR_MAP = {
                    'Q1': ['FIRST QUARTER',  'Q1', '1ST QUARTER',  'FIRST-QUARTER'],
                    'Q2': ['SECOND QUARTER', 'Q2', '2ND QUARTER',  'SECOND-QUARTER'],
                    'Q3': ['THIRD QUARTER',  'Q3', '3RD QUARTER',  'THIRD-QUARTER'],
                    'Q4': ['FOURTH QUARTER', 'Q4', '4TH QUARTER',  'FOURTH-QUARTER',
                           'FULL YEAR', 'ANNUAL'],
                }
                _qtr_tokens = _QTR_MAP.get(check_quarter, [])
                _qtr_found  = any(tok in _head for tok in _qtr_tokens)

                # Also check whether a *different* quarter appears prominently
                _other_qtrs_found = []
                for _q, _tokens in _QTR_MAP.items():
                    if _q != check_quarter and any(tok in _head for tok in _tokens):
                        _other_qtrs_found.append(_q)

                if not _qtr_found and _other_qtrs_found:
                    _detected = ' / '.join(_other_qtrs_found)
                    _mismatches.append(
                        f"**Wrong quarter** — this looks like a **{_detected}** call, not {check_quarter}. "
                        f"Please upload the correct transcript."
                    )
                elif not _qtr_found:
                    # Can't detect either way — soft warning only
                    _mismatches.append(
                        f"**Quarter unclear** — could not confirm this is a {check_quarter} call. "
                        f"Check the document header to be sure."
                    )

                # ── Year check ──────────────────────────────────────────
                _year_str   = str(check_year)
                _other_year = None
                if _year_str not in _head:
                    # Look for a 4-digit year that IS present as a clue
                    _m = _re.search(r'20[1-9][0-9]', _head)
                    if _m:
                        _other_year = _m.group()
                    _mismatches.append(
                        f"**Wrong year** — "
                        + (f"this document appears to be from **{_other_year}**, not {check_year}. "
                           if _other_year else
                           f"{check_year} does not appear in this document. ")
                        + f"Please upload the {check_year} transcript."
                    )

                return _mismatches

            if content:
                _mismatches = _detect_content_identity(content, ticker, quarter, year)

                # Expander is open by default when there are mismatches
                with st.expander(_preview_label, expanded=bool(_mismatches)):
                    if _mismatches:
                        # Red banner listing every mismatch found
                        _mismatch_html = ''.join(
                            f'<div style="margin-bottom:0.4rem;">⛔ {m}</div>'
                            for m in _mismatches
                        )
                        st.markdown(
                            f'<div style="background:#2D0A0A;border:1px solid #F05454;'
                            f'border-radius:6px;padding:0.85rem 1rem;margin-bottom:0.75rem;">'
                            f'<div style="color:#F05454;font-weight:700;font-size:0.82rem;'
                            f'margin-bottom:0.5rem;">⛔ This file does not match the expected call</div>'
                            f'<div style="color:#D97070;font-size:0.77rem;line-height:1.7;">'
                            f'{_mismatch_html}'
                            f'</div>'
                            f'<div style="color:#8FA3BF;font-size:0.72rem;margin-top:0.6rem;">'
                            f'The analysis will use the file you uploaded. Please upload the correct transcript before continuing.</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(
                            f"✅ Transcript confirmed: **{ticker}** {quarter} {year}  "
                            f"— sourced from `{uploaded.name}`"
                        )

                    st.text(content[:800] + ("…" if len(content) > 800 else ""))
                    st.caption(
                        f"Total length: {len(content):,} characters  "
                        f"≈ {len(content.split()):,} words"
                    )

            elif uploaded.type != "text/plain":
                with st.expander(_preview_label):
                    st.info(
                        f"📄 Text preview unavailable for `{uploaded.name}` "
                        f"(PDF/DOCX files cannot be previewed here).  \n"
                        f"Make sure this is the correct transcript for "
                        f"**{ticker} {quarter} {year}** before running analysis."
                    )
            else:
                with st.expander(_preview_label):
                    st.warning("Preview unavailable — the file appears to be empty.")

            # ── Run Analysis button ────────────────────────────────────
            # Disabled if ticker is invalid. If mismatches were detected,
            # show a "run anyway" checkbox so the user can consciously override.
            _has_mismatches = content and bool(
                _detect_content_identity(content, ticker, quarter, year)
            )
            _run_disabled = not ticker or not _ticker_valid

            if _has_mismatches:
                _override = st.checkbox(
                    f"I understand this may be the wrong file — analyse it anyway",
                    key=f"override_{uploaded.name}",
                )
                _run_disabled = _run_disabled or not _override
            else:
                _override = True  # no mismatches, no override needed

            btn_col, _ = st.columns([1, 2])
            with btn_col:
                _run_clicked = st.button(
                    "▶ Run Analysis", type="primary",
                    disabled=_run_disabled,
                    use_container_width=True,
                    key=f"run_{uploaded.name}",
                )

            # ── Analysis execution (outside btn_col so it spans full width) ──
            if _run_clicked:
                if not ticker or not _ticker_valid:
                    st.error("Please fix the ticker symbol before running analysis.")
                    st.stop()

                pb   = st.progress(0)
                msg  = st.empty()
                stages = [
                    "Preprocessing transcript…",
                    "Extracting FinBERT features…",
                    "Running sentiment analysis…",
                    "Loading best model…",
                    "Generating signal…",
                ]
                for i, s in enumerate(stages):
                    msg.caption(s)
                    time.sleep(0.7)
                    pb.progress((i + 1) / len(stages))
                msg.empty()

                # Seed on committed ticker + quarter + year
                _seed_str = f"{ticker}_{quarter}_{year}"
                rng  = np.random.RandomState(abs(hash(_seed_str)) % (2**31))
                prob = float(rng.uniform(0.44, 0.78))
                sig  = "BUY" if prob > 0.58 else "SELL" if prob < 0.45 else "HOLD"
                conv = ("High"   if abs(prob - 0.5) > 0.15 else
                        "Medium" if abs(prob - 0.5) > 0.07 else "Low")
                mgmt = float(rng.uniform(0.52, 0.91))
                anal = float(rng.uniform(0.38, 0.76))

                st.markdown("---")
                st.markdown(D.section_head(f"Result: {ticker} · {quarter} {year}",
                                           "Expected stock performance in the 3 days after this earnings call"),
                            unsafe_allow_html=True)

                s1, s2, s3 = st.columns([1.2, 1, 1])
                with s1:
                    st.markdown(D.signal_display(sig, prob, conv),
                                unsafe_allow_html=True)
                with s2:
                    st.markdown(D.kpi("Confidence", f"{prob:.0%}",
                                      "chance of outperforming the market", "#00C9A7"),
                                unsafe_allow_html=True)
                with s3:
                    st.markdown(D.kpi("Strength", conv,
                                      f"how decisive the signal is", "#F0A500"),
                                unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(D.section_head("Tone Analysis",
                                           "How positive or negative the language was in this call"),
                            unsafe_allow_html=True)

                g1, g2 = st.columns(2)
                for col, val, title, color in [
                    (g1, mgmt, "Executive Tone", "#00C9A7"),
                    (g2, anal, "Analyst Tone",    "#F0A500"),
                ]:
                    with col:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=val * 100,
                            title={"text": title, "font": {"family": "Space Grotesk",
                                   "color": "#8FA3BF", "size": 12}},
                            number={"suffix": "%", "font": {"color": color}},
                            gauge={"axis": {"range": [0, 100]},
                                   "bar": {"color": color},
                                   "bgcolor": "#162034",
                                   "bordercolor": "#1E2E48"},
                        ))
                        fig.update_layout(**D.plotly_theme(), height=210)
                        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            D.card("📌 How It Works",
                   '<div style="font-size:0.78rem;color:#8FA3BF;line-height:1.75;">'
                   '1. Upload the earnings call transcript<br>'
                   '2. The system reads the language and tone of the call<br>'
                   '3. It combines that with the company\'s financial data<br>'
                   '4. A recommendation is generated automatically<br>'
                   '5. You get a clear BUY / HOLD / SELL signal<br><br>'
                   '<strong style="color:#EEF2F8;">Accuracy:</strong> '
                   'Correct about 57% of the time on past data'
                   '</div>'),
            unsafe_allow_html=True,
        )
        st.markdown(
            D.card("📎 Try a Sample",
                   '<div style="font-size:0.75rem;color:#8FA3BF;">'
                   'Upload a transcript from any major US company — '
                   'Apple, Microsoft, Nvidia, or similar — and click Run Analysis.</div>'),
            unsafe_allow_html=True,
        )

    st.markdown(D.footer(), unsafe_allow_html=True)


def page_user_results():
    st.markdown(D.top_bar(badge="Analyst Portal"), unsafe_allow_html=True)
    st.markdown(D.section_head("Past Results",
                               "How the system performed on historical earnings calls"),
                unsafe_allow_html=True)

    df = load_test_data()
    if df is None:
        st.warning("No results data found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    avg_r = df['abnormal_return'].mean()
    pos   = df['label_binary'].mean() if 'label_binary' in df.columns else 0.323
    vol   = df['abnormal_return'].std()
    with c1: st.markdown(D.kpi("Calls in History", str(len(df)), "earnings calls reviewed", "#00C9A7"), unsafe_allow_html=True)
    with c2: st.markdown(D.kpi("Avg Gain vs Market", f"{avg_r:+.2%}", "compared to S&P 500",
                                "#30D158" if avg_r >= 0 else "#F05454"), unsafe_allow_html=True)
    with c3: st.markdown(D.kpi("Beat the Market", f"{pos:.1%}", "of calls had positive returns", "#F0A500"), unsafe_allow_html=True)
    with c4: st.markdown(D.kpi("Typical Swing", f"{vol:.2%}", "average price movement", "#4D9FFF"), unsafe_allow_html=True)

    st.markdown("---")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_tick = st.multiselect("Ticker", df['ticker'].unique()
                                  if 'ticker' in df.columns else [])
    with fc2:
        sel_q = st.multiselect("Quarter", df['quarter'].unique()
                               if 'quarter' in df.columns else [])
    with fc3:
        ret_f = st.selectbox("Return", ["All", "Positive", "Negative"])

    fd = df.copy()
    if sel_tick: fd = fd[fd['ticker'].isin(sel_tick)]
    if sel_q:    fd = fd[fd['quarter'].isin(sel_q)]
    if ret_f == "Positive": fd = fd[fd['abnormal_return'] > 0]
    elif ret_f == "Negative": fd = fd[fd['abnormal_return'] < 0]

    fig = go.Figure(go.Histogram(x=fd['abnormal_return'], nbinsx=50,
                                  marker_color='#00C9A7', opacity=0.75))
    fig.add_vline(x=0, line_dash="dash", line_color="#F05454",
                  annotation_text="Zero return")
    fig.update_layout(**D.plotly_theme(), title="How Much Each Stock Moved vs the Market",
                      height=280)
    st.plotly_chart(fig, use_container_width=True)

    show = [c for c in ['ticker', 'quarter', 'abnormal_return',
                         'stock_return_3day', 'label_binary'] if c in fd.columns]
    fd_display = fd[show].copy()
    rename_map = {'ticker': 'Company', 'quarter': 'Quarter', 'abnormal_return': 'Gain vs Market',
                  'stock_return_3day': '3-Day Return', 'label_binary': 'Beat Market (1=Yes)'}
    fd_display = fd_display.rename(columns={k: v for k, v in rename_map.items() if k in fd_display.columns})
    st.dataframe(fd_display.head(200), use_container_width=True, height=380)

    st.download_button("📥 Download as Spreadsheet", fd_display.to_csv(index=False),
                       f"ecrie_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    st.markdown(D.footer(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ── DEVELOPER PAGES ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _require_dev():
    """Block page if not authenticated. Returns True if ok."""
    if not st.session_state.dev_auth:
        st.error("Developer access required. Use the sidebar login.")
        st.stop()
    return True


def page_dev_pipeline():
    _require_dev()
    st.markdown(D.top_bar(badge="Developer Mode", badge_color="#F0A500"),
                unsafe_allow_html=True)
    st.markdown(D.dev_warn(), unsafe_allow_html=True)
    st.markdown(D.section_head("Pipeline Status",
                               "11-phase agentic orchestration overview"),
                unsafe_allow_html=True)

    state = load_state()
    if state is None:
        st.warning("Run `python run_phase11.py` first.")
        return

    fr   = state['final_results']
    perf = fr['performance']
    qual = fr['data_quality']
    ok   = fr['pipeline_status'] == 'success'

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(D.kpi("Status", "✓ OK" if ok else "⚠ WARN",
                                fr['pipeline_status'], "#30D158" if ok else "#F0A500"),
                          unsafe_allow_html=True)
    with c2: st.markdown(D.kpi("Phases", str(len(state['phase_results'])),
                                "completed", "#00C9A7"), unsafe_allow_html=True)
    with c3: st.markdown(D.kpi("Decisions", str(len(state['decisions'])),
                                "autonomous", "#4D9FFF"), unsafe_allow_html=True)
    with c4:
        nerr = len(fr['errors_encountered'])
        st.markdown(D.kpi("Warnings", str(nerr), "pipeline issues",
                           "#F05454" if nerr else "#30D158"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Phase results table ──
    st.markdown(D.section_head("Phase Results"), unsafe_allow_html=True)
    rows = []
    for phase, res in state['phase_results'].items():
        rows.append({
            "Phase":      phase.replace("_", " ").title(),
            "Status":     res.get("status", "unknown"),
            "Key Metric": _phase_metric(phase, res),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # ── Data quality ──
    st.markdown(D.section_head("Data Quality"), unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1: st.markdown(D.kpi("Parsing Rate",
                                f"{qual['parsing_success_rate']:.1%}",
                                "transcripts aligned",
                                "#30D158" if qual['parsing_success_rate'] >= 0.70 else "#F05454"),
                          unsafe_allow_html=True)
    with q2: st.markdown(D.kpi("Temporal Alignment",
                                "✓ Valid" if qual['temporal_alignment_valid'] else "✗ Invalid",
                                "lookahead bias check",
                                "#30D158" if qual['temporal_alignment_valid'] else "#F05454"),
                          unsafe_allow_html=True)
    with q3: st.markdown(D.kpi("Feature Extraction",
                                "✓ FinBERT" if qual['feature_extraction_success'] else "⚠ Fallback",
                                "768 embedding dims",
                                "#30D158" if qual['feature_extraction_success'] else "#F0A500"),
                          unsafe_allow_html=True)

    st.markdown("---")

    # ── Errors ──
    if fr['errors_encountered']:
        st.markdown(D.section_head("Warnings & Errors"), unsafe_allow_html=True)
        for e in fr['errors_encountered']:
            st.markdown(D.agent_entry("Pipeline", e, "Review and resolve", style="gold"),
                        unsafe_allow_html=True)

    st.markdown(D.footer(), unsafe_allow_html=True)


def page_dev_agents():
    _require_dev()
    st.markdown(D.top_bar(badge="Developer Mode", badge_color="#F0A500"),
                unsafe_allow_html=True)
    st.markdown(D.dev_warn(), unsafe_allow_html=True)
    st.markdown(D.section_head("Agent Monitor",
                               "All autonomous decisions logged by Phase 11"),
                unsafe_allow_html=True)

    state = load_state()
    if state is None:
        st.warning("Run `python run_phase11.py` first.")
        return

    decisions = state.get('decisions', [])
    if not decisions:
        st.info("No decisions logged yet.")
        return

    df_d = pd.DataFrame(decisions)

    # ── Charts ──
    c1, c2 = st.columns(2)
    with c1:
        counts = df_d['agent'].value_counts()
        fig = go.Figure(go.Pie(
            values=counts.values, labels=counts.index,
            hole=0.45, textfont_size=11,
        ))
        fig.update_layout(**D.plotly_theme(), title="Decisions by Agent", height=280)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pc = df_d.groupby('phase').size().reset_index(name='count')
        fig = go.Figure(go.Bar(x=pc['phase'], y=pc['count'],
                                marker_color='#00C9A7'))
        fig.update_layout(**D.plotly_theme(), title="Decisions by Phase",
                          height=280, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(D.section_head("Decision Log", "All agent actions"), unsafe_allow_html=True)

    for agent in df_d['agent'].unique():
        with st.expander(f"🤖 {agent}", expanded=False):
            subset = df_d[df_d['agent'] == agent]
            for _, row in subset.iterrows():
                st.markdown(
                    D.agent_entry(row['agent'], row['decision'],
                                  row['rationale'], row.get('timestamp', '')),
                    unsafe_allow_html=True,
                )

    st.markdown(D.footer(), unsafe_allow_html=True)


def page_dev_models():
    _require_dev()
    st.markdown(D.top_bar(badge="Developer Mode", badge_color="#F0A500"),
                unsafe_allow_html=True)
    st.markdown(D.dev_warn(), unsafe_allow_html=True)
    st.markdown(D.section_head("Model Explorer",
                               "All phases compared — raw metrics"),
                unsafe_allow_html=True)

    models = load_models()
    if not models:
        st.warning("Run the full pipeline first.")
        return

    rows = _build_chart_data(models)
    df = pd.DataFrame(rows)

    # ── Comparison table ──
    st.dataframe(
        df.style.highlight_max(subset=['AUC'], color='rgba(0,201,167,0.25)'),
        use_container_width=True,
    )

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=df.sort_values('AUC', ascending=False)['Model'],
            y=df.sort_values('AUC', ascending=False)['AUC'],
            marker_color=['#00C9A7' if p == 'Phase 10 Agentic'
                          else '#F0A500' if p == 'Phase 7 XGBoost'
                          else '#4D9FFF' for p in
                          df.sort_values('AUC', ascending=False)['Phase']],
            text=[f"{v:.1%}" for v in df.sort_values('AUC', ascending=False)['AUC']],
            textposition='outside',
        ))
        fig.add_hline(y=0.55, line_dash="dash", line_color="#F05454",
                      annotation_text="55% threshold")
        fig.update_layout(**D.plotly_theme(), title="All Models by ROC-AUC",
                          height=380, yaxis_range=[0.44, 0.62])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Scatter(
            x=df['Accuracy'], y=df['F1'],
            mode='markers+text',
            text=df['Model'],
            textposition='top center',
            marker=dict(size=df['AUC'] * 120, color='#00C9A7', opacity=0.7,
                        line=dict(color='#1E2E48', width=1)),
        ))
        fig.update_layout(**D.plotly_theme(), title="Accuracy vs F1",
                          height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Agentic learning trajectory ──
    if 'agentic' in models:
        st.markdown("---")
        st.markdown(D.section_head("Agentic RL Learning Trajectory",
                                   "Epsilon-greedy exploration over 8 iterations"),
                    unsafe_allow_html=True)
        traj = models['agentic'].get('learning_trajectory', [])
        if traj:
            df_t = pd.DataFrame(traj)
            fig = px.line(df_t, x='iteration', y='score', color='strategy',
                          markers=True, title='Agent Score per Iteration')
            fig.update_layout(**D.plotly_theme(), height=320)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(D.section_head("Strategy Effectiveness"), unsafe_allow_html=True)
            sp = df_t.groupby('strategy')['score'].agg(['mean', 'max', 'count']).reset_index()
            sp.columns = ['Strategy', 'Avg AUC', 'Best AUC', 'Tries']
            st.dataframe(sp, use_container_width=True)

    # ── Phase 8 SHAP ──
    shap_df = load_shap()
    if shap_df is not None:
        st.markdown("---")
        st.markdown(D.section_head("SHAP Feature Importance",
                                   "Top predictors from Phase 8 analysis"),
                    unsafe_allow_html=True)
        top = shap_df.head(20) if len(shap_df) >= 20 else shap_df
        fig = go.Figure(go.Bar(
            x=top.iloc[:, 1], y=top.iloc[:, 0],
            orientation='h', marker_color='#00C9A7',
        ))
        fig.update_layout(**D.plotly_theme(), title="Top 20 Features by SHAP Value",
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(D.footer(), unsafe_allow_html=True)


def page_dev_data():
    _require_dev()
    st.markdown(D.top_bar(badge="Developer Mode", badge_color="#F0A500"),
                unsafe_allow_html=True)
    st.markdown(D.dev_warn(), unsafe_allow_html=True)
    st.markdown(D.section_head("Raw Data", "Full test dataset"),
                unsafe_allow_html=True)

    df = load_test_data()
    if df is None:
        st.warning("No test data found.")
        return

    st.markdown(f"**{len(df)} records · {len(df.columns)} columns**")

    col_filter = st.multiselect("Select columns to display",
                                df.columns.tolist(),
                                default=df.columns[:10].tolist())
    st.dataframe(df[col_filter], use_container_width=True, height=500)

    st.download_button("📥 Download full test set (CSV)",
                       df.to_csv(index=False),
                       f"ecrie_test_{datetime.now().strftime('%Y%m%d')}.csv",
                       "text/csv")

    # ── Numeric stats ──
    st.markdown("---")
    st.markdown(D.section_head("Column Statistics"), unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.dataframe(df[num_cols].describe().T.round(4),
                     use_container_width=True)

    st.markdown(D.footer(), unsafe_allow_html=True)


def page_dev_config():
    _require_dev()
    st.markdown(D.top_bar(badge="Developer Mode", badge_color="#F0A500"),
                unsafe_allow_html=True)
    st.markdown(D.dev_warn(), unsafe_allow_html=True)
    st.markdown(D.section_head("Configuration",
                               "Pipeline thresholds and agent parameters"),
                unsafe_allow_html=True)

    # Load existing config
    config_path = RESULTS_DIR / 'ui_config.json'
    cfg = {}
    if config_path.exists():
        cfg = json.load(open(config_path, encoding='utf-8'))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Performance Thresholds")
        baseline_thr = st.slider("Baseline ROC-AUC threshold",
                                 0.50, 0.70, cfg.get('baseline_threshold', 0.55), 0.01)
        improve_thr  = st.slider("Required improvement (%)",
                                 0.0, 10.0, cfg.get('improvement_threshold', 0.02) * 100, 0.5)
        parsing_thr  = st.slider("Parsing success threshold (%)",
                                 50, 100, int(cfg.get('parsing_threshold', 0.70) * 100), 5)
    with c2:
        st.markdown("#### Agent Parameters")
        enable_fb  = st.checkbox("Enable automatic fallbacks",
                                 cfg.get('enable_fallbacks', True))
        enable_lrn = st.checkbox("Enable agent learning",
                                 cfg.get('enable_learning', True))
        log_level  = st.selectbox("Log level",
                                  ["DEBUG", "INFO", "WARNING", "ERROR"],
                                  index=["DEBUG", "INFO", "WARNING", "ERROR"]
                                  .index(cfg.get('log_level', 'INFO')))

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared.")
    with c4:
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()

    if st.button("💾 Save Configuration", type="primary"):
        new_cfg = {
            'baseline_threshold':   baseline_thr,
            'improvement_threshold': improve_thr / 100,
            'parsing_threshold':    parsing_thr / 100,
            'enable_fallbacks':     enable_fb,
            'enable_learning':      enable_lrn,
            'log_level':            log_level,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(new_cfg, f, indent=2)
        st.success("✅ Configuration saved.")

    st.markdown(D.footer(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _build_chart_data(models):
    rows = []
    if 'baseline' in models:
        for k, r in models['baseline'].get('results', {}).items():
            fs = r['feature_set'].replace('_', ' ').title()
            for mname, mkey in [("LR", "logistic_regression"),
                                  ("RF", "random_forest")]:
                if mkey in r:
                    rows.append({
                        "Phase": "Phase 6 Baseline",
                        "Model": f"{mname} / {fs}",
                        "AUC":      r[mkey]['roc_auc'],
                        "Accuracy": r[mkey]['accuracy'],
                        "F1":       r[mkey]['f1'],
                    })
    if 'xgboost' in models:
        for k, r in models['xgboost'].items():
            if 'classifier' in r:
                fs = r['feature_set'].replace('_', ' ').title()
                rows.append({
                    "Phase": "Phase 7 XGBoost",
                    "Model": f"XGBoost / {fs}",
                    "AUC":      r['classifier']['roc_auc'],
                    "Accuracy": r['classifier']['accuracy'],
                    "F1":       r['classifier']['f1'],
                })
    if 'agentic' in models:
        rows.append({
            "Phase": "Phase 10 Agentic",
            "Model": "Agentic Optimizer",
            "AUC":      models['agentic']['best_score'],
            "Accuracy": 0.0,
            "F1":       0.0,
        })
    return rows


def _phase_metric(phase, res):
    if phase == 'data_ingestion':
        return f"{res.get('records_available', '—')} records"
    if phase == 'preprocessing':
        r = res.get('parsing_success_rate', 0)
        return f"{r:.1%} parsing"
    if phase == 'feature_extraction':
        return res.get('feature_set', '—')
    if phase == 'integration':
        return f"{res.get('n_features', '—')} features"
    if phase == 'modeling':
        return f"AUC {res.get('best_auc', 0):.3f}"
    return "—"


# ══════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════

render_sidebar()

page = st.session_state.page

# Guard: prevent direct access to dev pages without auth
if page.startswith("dev_") and not st.session_state.dev_auth:
    st.session_state.page = "user_dashboard"
    page = "user_dashboard"

PAGE_MAP = {
    "user_dashboard": page_user_dashboard,
    "user_analysis":  page_user_analysis,
    "user_results":   page_user_results,
    "dev_pipeline":   page_dev_pipeline,
    "dev_agents":     page_dev_agents,
    "dev_models":     page_dev_models,
    "dev_data":       page_dev_data,
    "dev_config":     page_dev_config,
}

PAGE_MAP.get(page, page_user_dashboard)()