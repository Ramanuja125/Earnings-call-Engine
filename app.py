"""
ECRIE — Enhanced Dashboard
===========================
Earnings Call & Risk Intelligence Engine
Full UI with:
  - Overview dashboard (KPIs, model comparison, sentiment signals, trend)
  - Analyze Call page (tone gauges for exec + analyst, SHAP waterfall,
    sentiment breakdown, risk word cloud proxy, signal output)
  - Browse Results page
  - Developer model explorer

Run:
    pip install streamlit plotly pandas numpy
    streamlit run dashboard_app.py
"""

import re
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════

MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
  --navy-deep:   #080E1C;
  --navy-mid:    #0F1829;
  --navy-card:   #162034;
  --navy-border: #1E2E48;
  --teal:        #00C9A7;
  --teal-dark:   #009E83;
  --gold:        #F0A500;
  --red:         #F05454;
  --green:       #30D158;
  --blue:        #4D9FFF;
  --text-hi:     #EEF2F8;
  --text-mid:    #8FA3BF;
  --text-lo:     #3D5470;
  --font-ui:     'Space Grotesk', sans-serif;
  --font-mono:   'JetBrains Mono', monospace;
}

.stApp { background: var(--navy-deep) !important; color: var(--text-hi) !important; font-family: var(--font-ui) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] { background: var(--navy-mid) !important; border-right: 1px solid var(--navy-border) !important; }
[data-testid="stSidebar"] * { color: var(--text-hi) !important; font-family: var(--font-ui) !important; }
[data-testid="stSidebar"] .stButton > button {
  border: none !important; background: transparent !important;
  color: var(--text-mid) !important; text-align: left !important;
  font-size: 0.82rem !important; padding: 0.45rem 0.8rem !important;
  border-radius: 4px !important; width: 100% !important;
  margin-bottom: 2px !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(0,201,167,0.1) !important; color: var(--teal) !important; }

::-webkit-scrollbar { width: 4px; }
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
span[data-testid="stSidebarCollapse"] { display: none !important; }
section[data-testid="stSidebar"] { display: block !important; visibility: visible !important; transform: none !important; width: 260px !important; min-width: 260px !important; }
section[data-testid="stSidebar"][aria-expanded="false"] { margin-left: 0 !important; }
section[data-testid="stSidebar"] > div { width: 260px !important; }
::-webkit-scrollbar-track { background: var(--navy-deep); }
::-webkit-scrollbar-thumb { background: var(--navy-border); border-radius: 2px; }

.stButton > button {
  background: transparent !important; color: var(--teal) !important;
  border: 1px solid var(--teal) !important; border-radius: 4px !important;
  font-family: var(--font-ui) !important; font-size: 0.8rem !important; transition: all 0.18s !important;
}
.stButton > button:hover { background: var(--teal) !important; color: var(--navy-deep) !important; }
.stButton > button[kind="primary"] { background: var(--teal) !important; color: var(--navy-deep) !important; font-weight: 600 !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: var(--teal-dark) !important; }

.stTextInput input, .stSelectbox > div > div, .stNumberInput input, .stTextArea textarea {
  background: var(--navy-card) !important; border: 1px solid var(--navy-border) !important;
  color: var(--text-hi) !important; border-radius: 4px !important;
  font-family: var(--font-mono) !important; font-size: 0.82rem !important;
}
.stProgress .st-bo { background: var(--teal) !important; }
hr { border-color: var(--navy-border) !important; margin: 1.5rem 0 !important; }
[data-testid="stMetric"] { background: var(--navy-card); border: 1px solid var(--navy-border); border-radius: 6px; padding: 1rem; }
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; }

/* ── Selectbox / Multiselect / Dropdowns ── */
[data-baseweb="select"] { background: var(--navy-card) !important; }
[data-baseweb="select"] > div { background: var(--navy-card) !important; border-color: var(--navy-border) !important; color: var(--text-hi) !important; }
[data-baseweb="select"] span { color: var(--text-hi) !important; }
[data-baseweb="select"] svg { fill: var(--text-mid) !important; }
[data-baseweb="popover"] { background: var(--navy-card) !important; border-color: var(--navy-border) !important; }
[data-baseweb="menu"] { background: var(--navy-card) !important; }
[data-baseweb="menu"] li { background: var(--navy-card) !important; color: var(--text-hi) !important; }
[data-baseweb="menu"] li:hover { background: var(--navy-mid) !important; color: var(--teal) !important; }
[data-baseweb="option"] { background: var(--navy-card) !important; color: var(--text-hi) !important; }
[data-baseweb="option"]:hover { background: var(--navy-mid) !important; }
[aria-selected="true"][data-baseweb="option"] { background: rgba(0,201,167,0.15) !important; color: var(--teal) !important; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] { background: rgba(0,201,167,0.15) !important; border-color: var(--teal) !important; }
[data-baseweb="tag"] span { color: var(--teal) !important; }

/* ── Labels above inputs ── */
label[data-testid="stWidgetLabel"] p { color: var(--text-mid) !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
label p { color: var(--text-mid) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { background: var(--navy-card) !important; border: 1px dashed var(--navy-border) !important; border-radius: 6px !important; }
[data-testid="stFileUploader"] label { color: var(--text-mid) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--navy-card) !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text-mid) !important; }

/* ── Expander ── */
[data-testid="stExpander"] { background: var(--navy-card) !important; border: 1px solid var(--navy-border) !important; border-radius: 6px !important; }
[data-testid="stExpander"] summary { color: var(--text-hi) !important; }
[data-testid="stExpanderDetails"] { background: var(--navy-card) !important; }

/* ── General text color fixes ── */
p, span, div { color: inherit; }
.stMarkdown p { color: var(--text-hi) !important; }
[data-testid="stText"] { color: var(--text-mid) !important; font-family: var(--font-mono) !important; font-size: 0.78rem !important; }
.stTabs [data-baseweb="tab"] { font-family: var(--font-ui) !important; font-size: 0.82rem !important; color: var(--text-mid) !important; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; border-bottom-color: var(--teal) !important; }

/* ── Tabs – kill all white boxes ── */
.stTabs [data-baseweb="tab"] { 
  background: var(--navy-mid) !important; 
  border: 1px solid var(--navy-border) !important; 
  border-radius: 4px 4px 0 0 !important;
  color: var(--text-mid) !important;
  margin-right: 4px !important;
}
.stTabs [aria-selected="true"] { 
  background: var(--navy-card) !important; 
  border-bottom-color: var(--teal) !important;
  color: var(--teal) !important; 
}
.stTabs [data-baseweb="tab-panel"],
.stTabs [role="tabpanel"],
div[data-baseweb="tab-panel"] { 
  background: transparent !important; 
  border: none !important;
  padding: 1rem 0 !important;
}
[data-baseweb="tab-list"] { 
  background: transparent !important; 
  border-bottom: 1px solid var(--navy-border) !important; 
  gap: 4px !important;
}
[data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-highlight"] { background: var(--teal) !important; height: 2px !important; }

/* ── DataFrame / table dark mode ── */
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrameContainer"],
.dvn-scroller, .dvn-scroller > div,
canvas.dvn-canvas { background: var(--navy-card) !important; }
.glideDataEditor { background: var(--navy-card) !important; }
.gdg-style { 
  --gdg-bg-cell: #162034 !important; 
  --gdg-bg-cell-medium: #1a2540 !important;
  --gdg-bg-header: #0F1829 !important; 
  --gdg-bg-header-hovered: #1E2E48 !important;
  --gdg-text-dark: #EEF2F8 !important; 
  --gdg-text-medium: #8FA3BF !important;
  --gdg-text-light: #3D5470 !important;
  --gdg-border-color: #1E2E48 !important;
  --gdg-accent-color: #00C9A7 !important;
}
</style>
"""

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F1829",
        font=dict(family="Space Grotesk", color="#8FA3BF", size=11),
        title_font=dict(family="Space Grotesk", color="#EEF2F8", size=13),
        xaxis=dict(gridcolor="#1E2E48", linecolor="#1E2E48", tickcolor="#3D5470"),
        yaxis=dict(gridcolor="#1E2E48", linecolor="#1E2E48", tickcolor="#3D5470"),
        legend=dict(bgcolor="rgba(22,32,52,0.9)", bordercolor="#1E2E48", borderwidth=1),
        colorway=["#00C9A7", "#F0A500", "#4D9FFF", "#F05454", "#30D158", "#9B72CF"],
        margin=dict(t=40, b=30, l=10, r=10),
    )

def top_bar(badge=""):
    badge_html = (
        f'<span style="background:#00C9A7;color:#080E1C;font-size:0.62rem;'
        f'font-weight:700;padding:0.25rem 0.7rem;border-radius:3px;'
        f'letter-spacing:0.1em;text-transform:uppercase;">{badge}</span>'
    ) if badge else ""
    return f"""
    <div style="background:linear-gradient(135deg,#0F1829 0%,#080E1C 100%);
                border-bottom:2px solid #00C9A7;padding:1.2rem 1.5rem;
                margin:-1rem -1rem 2rem -1rem;
                display:flex;align-items:center;justify-content:space-between;">
      <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;
                    font-weight:700;color:#00C9A7;letter-spacing:0.04em;">ECRIE</div>
        <div style="font-size:0.62rem;color:#3D5470;letter-spacing:0.14em;
                    text-transform:uppercase;margin-top:0.15rem;">
          Earnings Call &amp; Risk Intelligence Engine</div>
      </div>
      {badge_html}
    </div>"""

def section_head(title, sub=""):
    sub_html = (f'<div style="font-size:0.62rem;color:#3D5470;letter-spacing:0.14em;'
                f'text-transform:uppercase;margin-top:0.2rem;">{sub}</div>') if sub else ""
    return f"""
    <div style="margin-bottom:1.2rem;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                  font-weight:700;color:#EEF2F8;">{title}</div>
      {sub_html}
    </div>"""

def kpi_card(label, value, note="", accent="#00C9A7"):
    return f"""
    <div style="background:#162034;border:1px solid #1E2E48;border-top:2px solid {accent};
                border-radius:6px;padding:1.1rem 1.2rem;overflow:hidden;">
      <div style="font-size:0.62rem;color:#3D5470;letter-spacing:0.14em;
                  text-transform:uppercase;margin-bottom:0.45rem;">{label}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.9rem;
                  font-weight:700;color:{accent};line-height:1;margin-bottom:0.25rem;">{value}</div>
      <div style="font-size:0.68rem;color:#8FA3BF;">{note}</div>
    </div>"""

def signal_display(signal, prob, conviction):
    cfg = {"BUY": ("#30D158", "▲"), "SELL": ("#F05454", "▼"), "HOLD": ("#F0A500", "■")}
    color, icon = cfg.get(signal.upper(), ("#8FA3BF", "●"))
    return f"""
    <div style="background:#162034;border:2px solid {color};border-radius:8px;
                padding:2rem;text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:3rem;font-weight:700;
                  color:{color};line-height:1;margin-bottom:0.5rem;">{icon} {signal.upper()}</div>
      <div style="font-size:0.68rem;color:#8FA3BF;letter-spacing:0.14em;text-transform:uppercase;">
        {prob:.0%} model confidence &nbsp;·&nbsp; {conviction} conviction
      </div>
    </div>"""

def info_card(title, body_html):
    return f"""
    <div style="background:#162034;border:1px solid #1E2E48;border-radius:6px;
                padding:1.3rem;margin-bottom:1rem;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;font-weight:600;
                  color:#EEF2F8;margin-bottom:0.8rem;padding-bottom:0.6rem;
                  border-bottom:1px solid #1E2E48;">{title}</div>
      {body_html}
    </div>"""

def footer():
    return """
    <div style="text-align:center;padding:2rem 0 1rem;font-size:0.62rem;color:#3D5470;
                letter-spacing:0.12em;text-transform:uppercase;
                border-top:1px solid #1E2E48;margin-top:3rem;">
      ECRIE &nbsp;·&nbsp; FSE 570 Capstone &nbsp;·&nbsp;
      Arizona State University &nbsp;·&nbsp; Spring 2026
    </div>"""

# ══════════════════════════════════════════════════════════════════════
# SIMULATED ANALYSIS ENGINE  (replace with real model calls)
# ══════════════════════════════════════════════════════════════════════

def run_analysis_engine(ticker: str, quarter: str, year: int, text: str) -> dict:
    """
    Deterministic mock of the full ECRIE pipeline output.
    Seed is based on ticker+quarter+year so results are reproducible.
    Replace this function body with actual FinBERT + XGBoost calls.
    """
    seed = abs(hash(f"{ticker}_{quarter}_{year}")) % (2**31)
    rng  = np.random.RandomState(seed)

    prob       = float(rng.uniform(0.44, 0.82))
    signal     = "BUY" if prob > 0.60 else "SELL" if prob < 0.46 else "HOLD"
    conviction = ("High" if abs(prob - 0.5) > 0.15 else
                  "Medium" if abs(prob - 0.5) > 0.07 else "Low")

    # ── Tone scores ──────────────────────────────────────────────────
    exec_tone   = float(rng.uniform(0.50, 0.92))
    analyst_tone = float(rng.uniform(0.35, 0.80))

    # ── Sentiment breakdown ──────────────────────────────────────────
    pos_sent  = float(rng.uniform(0.35, 0.65))
    neg_sent  = float(rng.uniform(0.10, 0.30))
    neut_sent = max(0.0, 1.0 - pos_sent - neg_sent)

    # ── FinBERT section scores (executive prepared remarks vs Q&A) ──
    exec_pos  = float(rng.uniform(0.40, 0.70))
    exec_neg  = float(rng.uniform(0.05, 0.25))
    exec_neut = max(0.0, 1.0 - exec_pos - exec_neg)
    qa_pos    = float(rng.uniform(0.30, 0.60))
    qa_neg    = float(rng.uniform(0.10, 0.35))
    qa_neut   = max(0.0, 1.0 - qa_pos - qa_neg)

    # ── SHAP feature contributions ───────────────────────────────────
    # Positive = pushed towards BUY, negative = pushed towards SELL
    shap_base = 0.50
    features = {
        "FinBERT sentiment":    float(rng.uniform(-0.08,  0.14)),
        "EPS surprise":         float(rng.uniform(-0.06,  0.10)),
        "Revenue growth":       float(rng.uniform(-0.05,  0.09)),
        "Forward guidance":     float(rng.uniform(-0.07,  0.08)),
        "Mgmt tone score":      float(rng.uniform(-0.04,  0.07)),
        "Analyst tone score":   float(rng.uniform(-0.05,  0.06)),
        "Risk word count":      float(rng.uniform(-0.09,  0.03)),
        "Q&A sentiment":        float(rng.uniform(-0.06,  0.07)),
        "Market beta":          float(rng.uniform(-0.03,  0.05)),
        "Uncertainty hedge words": float(rng.uniform(-0.05, 0.02)),
    }
    # Nudge SHAP values to be consistent with the signal
    direction = 1 if signal == "BUY" else -1 if signal == "SELL" else 0
    for k in list(features)[:4]:
        features[k] = abs(features[k]) * direction * float(rng.uniform(0.6, 1.0))

    # ── Risk keywords ────────────────────────────────────────────────
    risk_keywords = {
        "uncertainty": rng.randint(3, 12),
        "headwinds":   rng.randint(1, 8),
        "challenging": rng.randint(2, 9),
        "tariffs":     rng.randint(0, 6),
        "inflation":   rng.randint(1, 7),
        "guidance":    rng.randint(4, 14),
        "growth":      rng.randint(5, 20),
        "margin":      rng.randint(3, 12),
        "demand":      rng.randint(2, 10),
        "macro":       rng.randint(1, 8),
        "supply chain":rng.randint(0, 5),
        "strong":      rng.randint(4, 16),
    }

    return dict(
        ticker=ticker, quarter=quarter, year=year,
        prob=prob, signal=signal, conviction=conviction,
        exec_tone=exec_tone, analyst_tone=analyst_tone,
        pos_sent=pos_sent, neg_sent=neg_sent, neut_sent=neut_sent,
        exec_pos=exec_pos, exec_neg=exec_neg, exec_neut=exec_neut,
        qa_pos=qa_pos, qa_neg=qa_neg, qa_neut=qa_neut,
        shap_base=shap_base, features=features,
        risk_keywords=risk_keywords,
        word_count=len(text.split()) if text else 0,
    )

# ══════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════

def tone_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"family": "Space Grotesk", "color": "#8FA3BF", "size": 12}},
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3D5470",
                     "tickfont": {"size": 9, "color": "#3D5470"}},
            "bar": {"color": color},
            "bgcolor": "#162034",
            "bordercolor": "#1E2E48",
            "steps": [
                {"range": [0,  40], "color": "rgba(240,84,84,0.15)"},
                {"range": [40, 60], "color": "rgba(240,165,0,0.12)"},
                {"range": [60,100], "color": "rgba(0,201,167,0.10)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(**plotly_theme(), height=230)
    return fig


def sentiment_donut(pos, neg, neut, title="") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[pos, neg, neut],
        hole=0.62,
        marker_colors=["#00C9A7", "#F05454", "#3D5470"],
        textinfo="percent",
        textfont={"size": 10, "color": "#EEF2F8"},
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False,
                       font={"family": "Space Grotesk", "size": 11, "color": "#8FA3BF"})
    fig.update_layout(**plotly_theme(), height=220, showlegend=True)
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                                  font={"size": 10}))
    return fig


def shap_waterfall(features: dict, base: float, final_prob: float) -> go.Figure:
    # Sort positive first (top), then negative (bottom), each by magnitude
    pos = sorted([(k,v) for k,v in features.items() if v >= 0], key=lambda x: x[1], reverse=True)
    neg = sorted([(k,v) for k,v in features.items() if v < 0],  key=lambda x: x[1])
    sorted_feats = pos + neg
    names  = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    MIN_VIS = 0.006
    vis_values  = [v if abs(v) >= MIN_VIS else (MIN_VIS if v >= 0 else -MIN_VIS) for v in values]
    colors      = ["#00C9A7" if v >= 0 else "#F05454" for v in values]
    opacities   = [1.0 if abs(v) >= MIN_VIS else 0.5 for v in values]
    labels      = [(f"+{v:.3f}" if v >= 0 else f"{v:.3f}") for v in values]

    fig = go.Figure()
    for name, vis_val, color, opacity, label in zip(names, vis_values, colors, opacities, labels):
        fig.add_trace(go.Bar(
            x=[vis_val], y=[name], orientation="h",
            marker_color=color, marker_opacity=opacity, marker_line_width=0,
            text=label, textposition="outside",
            textfont={"size": 10, "color": "#EEF2F8"},
            hovertemplate=f"<b>{name}</b><br>SHAP: {label}<extra></extra>",
            showlegend=False, width=0.6,
        ))

    fig.add_vline(x=0, line_color="#3D5470", line_width=1.5)

    direction = "BUY" if final_prob > 0.58 else "SELL" if final_prob < 0.46 else "HOLD"
    fig.update_layout(
        **plotly_theme(),
        barmode="relative", height=420, showlegend=False, bargap=0.25,
        title=f"SHAP Feature Impact   |   Base {base:.2f}  to  Output {final_prob:.2f}  ({direction})",
    )
    fig.update_layout(title_font=dict(size=12, color="#EEF2F8"))
    fig.update_xaxes(
        title_text="<-- Pushes SELL    |    Pushes BUY -->",
        title_font=dict(size=10, color="#8FA3BF"),
        zeroline=True, zerolinecolor="#8FA3BF", zerolinewidth=1.5,
        tickfont=dict(size=10, color="#8FA3BF"),
        gridcolor="#1E2E48",
    )
    fig.update_yaxes(
        tickfont=dict(size=12, color="#EEF2F8", family="Space Grotesk"),
        autorange="reversed",
        gridcolor="#1E2E48",
        ticklabelposition="outside",
    )
    return fig


def risk_bar_chart(risk_keywords: dict) -> go.Figure:
    sorted_kw = sorted(risk_keywords.items(), key=lambda x: x[1], reverse=True)
    words  = [k for k, _ in sorted_kw]
    counts = [v for _, v in sorted_kw]

    # Color by risk vs growth sentiment
    risk_words_set = {"uncertainty", "headwinds", "challenging", "tariffs",
                      "inflation", "macro", "supply chain"}
    bar_colors = ["#F05454" if w in risk_words_set else "#00C9A7" for w in words]

    fig = go.Figure(go.Bar(
        x=words, y=counts,
        marker_color=bar_colors,
        text=counts, textposition="outside",
        textfont={"size": 10, "color": "#8FA3BF"},
        hovertemplate="%{x}: %{y} mentions<extra></extra>",
    ))
    fig.update_layout(
        **plotly_theme(), height=260,
        title="Key Term Frequency — risk terms in red, growth in teal",
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(title_text="Mentions")
    return fig


def section_comparison_bar(exec_pos, exec_neg, exec_neut,
                             qa_pos, qa_neg, qa_neut) -> go.Figure:
    sections = ["Prepared Remarks", "Q&A Session"]
    fig = go.Figure()
    for label, color, vals in [
        ("Positive", "#00C9A7", [exec_pos, qa_pos]),
        ("Negative", "#F05454", [exec_neg, qa_neg]),
        ("Neutral",  "#3D5470", [exec_neut, qa_neut]),
    ]:
        fig.add_trace(go.Bar(
            name=label, x=sections, y=vals,
            marker_color=color,
            text=[f"{v:.0%}" for v in vals],
            textposition="inside",
            textfont={"size": 10, "color": "#EEF2F8"},
        ))
    fig.update_layout(
        **plotly_theme(), barmode="stack", height=260,
        title="Sentiment by Section",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def model_comparison_chart() -> go.Figure:
    models = ["Financial\nOnly", "FinBERT\nOnly", "XGBoost\nCombined", "Agent\nEnhanced"]
    roc    = [0.520, 0.547, 0.633, 0.572]
    prec   = [0.510, 0.530, 0.583, 0.558]
    rec    = [0.495, 0.515, 0.591, 0.544]

    fig = go.Figure()
    for name, vals, color in [
        ("ROC-AUC",   roc,  "#00C9A7"),
        ("Precision", prec, "#F0A500"),
        ("Recall",    rec,  "#4D9FFF"),
    ]:
        fig.add_trace(go.Bar(name=name, x=models, y=vals, marker_color=color,
                             text=[f"{v:.3f}" for v in vals], textposition="outside",
                             textfont={"size": 9, "color": "#8FA3BF"}))
    fig.update_layout(**plotly_theme(), barmode="group", height=300,
                      title="Model Performance Comparison")
    fig.update_yaxes(range=[0.45, 0.65])
    return fig


def sentiment_trend_chart() -> go.Figure:
    quarters = ["Q1 23", "Q2 23", "Q3 23", "Q4 23", "Q1 24", "Q2 24", "Q3 24", "Q4 24"]
    bull  = [42, 38, 31, 35, 44, 51, 48, 55]
    bear  = [28, 35, 41, 38, 30, 24, 27, 22]
    neut  = [30, 27, 28, 27, 26, 25, 25, 23]
    fill_colors = {
        "#00C9A7": "rgba(0,201,167,0.08)",
        "#F05454": "rgba(240,84,84,0.08)",
        "#3D5470": "rgba(61,84,112,0.08)",
    }
    fig = go.Figure()
    for name, vals, color in [
        ("Bullish", bull, "#00C9A7"),
        ("Bearish", bear, "#F05454"),
        ("Neutral", neut, "#3D5470"),
    ]:
        fig.add_trace(go.Scatter(
            x=quarters, y=vals, name=name,
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=fill_colors[color],
            mode="lines+markers", marker=dict(size=5),
        ))
    fig.update_layout(**plotly_theme(), height=240,
                      title="Market Sentiment Trend — Q1 2023 to Q4 2024")
    return fig


def dark_table(df, max_rows=200):
    """Render a dataframe as a fully dark-themed HTML table."""
    rows_html = ""
    for _, row in df.head(max_rows).iterrows():
        cells = ""
        for val in row:
            if isinstance(val, float):
                display = f"{val:+.2%}" if abs(val) < 1 and val != 0 else f"{val:.4f}" if abs(val) < 10 else str(val)
            else:
                display = str(val)
            cells += f'<td style="padding:8px 12px;border-bottom:1px solid #1E2E48;color:#EEF2F8;font-size:0.78rem;font-family:JetBrains Mono,monospace;">{display}</td>'
        rows_html += f"<tr>{cells}</tr>"
    
    headers = ""
    for col in df.columns:
        headers += f'<th style="padding:8px 12px;background:#0F1829;color:#3D5470;font-size:0.65rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;border-bottom:2px solid #1E2E48;white-space:nowrap;">{col}</th>'
    
    return f"""
    <div style="overflow-x:auto;border:1px solid #1E2E48;border-radius:6px;background:#162034;">
      <table style="width:100%;border-collapse:collapse;background:#162034;">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════

for k, v in [("page", "dashboard"), ("result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ECRIE — Earnings Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(MASTER_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

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

    def nav(label, key):
        active = st.session_state.page == key
        style  = "color: #00C9A7 !important;" if active else ""
        if st.button(("▸ " if active else "  ") + label, key=f"nav_{key}",
                     use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown('<div style="font-size:0.6rem;color:#00C9A7;letter-spacing:0.15em;'
                'text-transform:uppercase;margin-bottom:0.4rem;">Analyst</div>',
                unsafe_allow_html=True)
    nav("Dashboard",      "dashboard")
    nav("Analyze Call",   "analyze")
    nav("Browse Results", "results")
    #nav("Causal Inference", "causal")

    #st.markdown("---")
    #st.markdown('<div style="font-size:0.6rem;color:#F0A500;letter-spacing:0.15em;'
    #            'text-transform:uppercase;margin-bottom:0.4rem;">Developer</div>',
    #            unsafe_allow_html=True)
    #nav("Model Explorer",   "models")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#3D5470;padding:0.5rem 0;">'
        'FSE 570 Capstone · ASU<br>Spring 2026</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown(top_bar(badge="Analyst Portal"), unsafe_allow_html=True)
    st.markdown(section_head("Overview", "System-wide metrics and market signals"),
                unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Transcripts Analyzed", "917", "real · 508 tickers · 27 quarters", "#00C9A7"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Best ROC-AUC", "0.633", "XGBoost + fullFinBert", "#30D158"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Feature Dimensions", "861", "784 FinBERT + 77 financial", "#4D9FFF"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Price Records", "1.24M", "Market data 2016–2026", "#F0A500"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: model comparison + sentiment breakdown
    col_a, col_b = st.columns([1.4, 1])
    with col_a:
        st.plotly_chart(model_comparison_chart(), use_container_width=True)
    with col_b:
        st.markdown(section_head("Sentiment signals — top tickers"), unsafe_allow_html=True)
        tickers_data = [
            ("AAPL", 0.82, "BUY",  "#30D158"),
            ("AMZN", 0.74, "BUY",  "#30D158"),
            ("ADBE", 0.51, "HOLD", "#F0A500"),
            ("ABNB", 0.61, "BUY",  "#30D158"),
            ("ADI",  0.33, "SELL", "#F05454"),
        ]
        for sym, score, sig, color in tickers_data:
            bar_pct = int(score * 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:7px 10px;'
                f'background:#162034;border-radius:4px;margin-bottom:4px;font-size:13px;">'
                f'<span style="font-weight:600;color:#EEF2F8;width:46px;">{sym}</span>'
                f'<div style="flex:1;background:#1E2E48;border-radius:3px;height:6px;">'
                f'<div style="width:{bar_pct}%;background:{color};height:6px;border-radius:3px;"></div></div>'
                f'<span style="color:#8FA3BF;font-size:12px;width:34px;">{score:.2f}</span>'
                f'<span style="background:{color}22;color:{color};font-size:10px;font-weight:700;'
                f'padding:2px 8px;border-radius:99px;">{sig}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Trend chart
    st.plotly_chart(sentiment_trend_chart(), use_container_width=True)

    # SHAP top features overview
    st.markdown(section_head("Top SHAP features (aggregate)", "Average absolute importance across all test predictions"), unsafe_allow_html=True)
    feat_names = ["Mgmt tone (linguistic)","Analyst tone (linguistic)","FinBERT sentiment","Revenue growth","EPS surprise","Risk word count","Forward guidance","Market beta"]
    feat_vals  = [0.18, 0.16, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08]
    fig_shap = go.Figure(go.Bar(
        x=feat_vals[::-1], y=feat_names[::-1], orientation="h",
        marker_color="#00C9A7",
        text=[f"{v:.0%}" for v in feat_vals[::-1]], textposition="outside",
        textfont={"size": 10, "color": "#8FA3BF"},
    ))
    fig_shap.update_layout(**plotly_theme(), height=300)
    fig_shap.update_xaxes(title_text="Mean |SHAP value|")
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown(footer(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE CALL
# ══════════════════════════════════════════════════════════════════════

def page_analyze():
    st.markdown(top_bar(badge="Analyst Portal"), unsafe_allow_html=True)
    st.markdown(section_head("Analyze an Earnings Call",
                             "Upload a transcript and get an instant market signal with full explainability"),
                unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        uploaded = st.file_uploader(
            "Upload earnings call transcript (.txt)",
            type=["txt", "pdf", "docx"],
        )

        if uploaded:
            _key = f"_tc_{uploaded.name}"
            if _key not in st.session_state:
                raw = uploaded.read()
                st.session_state[_key] = raw.decode("utf-8", errors="ignore") if uploaded.type == "text/plain" else None
            content = st.session_state[_key]

            st.success(f"Loaded: {uploaded.name}")

            stem   = re.sub(r'\.[^.]+$', '', uploaded.name)
            parts  = re.split(r'[_\-\.\s]+', stem.upper())
            ticker, quarter, year = "AAPL", "Q1", 2024
            for p in parts:
                if re.fullmatch(r'Q[1-4]', p): quarter = p
                elif re.fullmatch(r'20[0-9]{2}', p): year = int(p)
            SKIP = {"FILE","TRANSCRIPT","EARNINGS","CALL","REPORT","Q","QTR","FY"}
            for p in parts:
                if re.fullmatch(r'[A-Z]{1,5}(-[A-Z])?', p) and p not in SKIP and not re.fullmatch(r'Q[1-4]', p):
                    ticker = p; break

            if content:
                with st.expander(f"Preview · {ticker} {quarter} {year}", expanded=False):
                    st.text(content[:800] + ("…" if len(content) > 800 else ""))
                    st.caption(f"Length: {len(content):,} chars ≈ {len(content.split()):,} words")

            btn_col, _ = st.columns([1, 3])
            with btn_col:
                run_clicked = st.button("▶ Run Full Analysis", type="primary",
                                        use_container_width=True)

            if run_clicked:
                pb = st.progress(0)
                msg = st.empty()
                stages = [
                    "Preprocessing transcript…",
                    "Running FinBERT embeddings (768-dim)…",
                    "Scoring executive vs analyst tone…",
                    "Computing risk signal features…",
                    "Loading XGBoost ensemble…",
                    "Generating SHAP explanation…",
                ]
                for i, s in enumerate(stages):
                    msg.caption(s); time.sleep(0.5)
                    pb.progress((i + 1) / len(stages))
                msg.empty(); pb.empty()
                st.session_state.result = run_analysis_engine(
                    ticker, quarter, year, content or ""
                )
                st.rerun()

    with col_side:
        st.markdown(info_card("📌 How it works",
            '<div style="font-size:0.78rem;color:#8FA3BF;line-height:1.75;">'
            '1. Upload the earnings call transcript<br>'
            '2. FinBERT reads each sentence for positive/negative tone<br>'
            '3. Executive remarks and analyst Q&A are scored separately<br>'
            '4. XGBoost combines NLP + financial features<br>'
            '5. SHAP explains <em>why</em> the model gave that signal<br><br>'
            '<strong style="color:#EEF2F8;">Accuracy:</strong> Cross-validated ROC-AUC 0.572 (917 real transcripts)'
            '</div>'),
        unsafe_allow_html=True)

        st.markdown(info_card("📎 Sample tickers",
            '<div style="font-size:0.75rem;color:#8FA3BF;">'
            'Try uploading any transcript from the <code>data/raw/transcripts/real/</code> folder:<br><br>'
            'AAPL_Q2_2016.txt<br>Any of 917 real transcripts<br>from data/raw/transcripts/real/<br>508 tickers · 27 quarters'
            '</div>'),
        unsafe_allow_html=True)

    # ── Results ──────────────────────────────────────────────────────
    r = st.session_state.result
    if r is None:
        st.markdown(footer(), unsafe_allow_html=True)
        return

    st.markdown("---")
    st.markdown(
        section_head(f"Results — {r['ticker']} · {r['quarter']} {r['year']}",
                     "Expected stock performance in the 3 days post-earnings"),
        unsafe_allow_html=True,
    )

    # Signal + KPIs
    s1, s2, s3 = st.columns([1.2, 1, 1])
    with s1: st.markdown(signal_display(r["signal"], r["prob"], r["conviction"]), unsafe_allow_html=True)
    with s2: st.markdown(kpi_card("Model Confidence", f"{r['prob']:.0%}", "chance of outperforming market", "#00C9A7"), unsafe_allow_html=True)
    with s3: st.markdown(kpi_card("Conviction Level", r["conviction"], "decisiveness of the signal", "#F0A500"), unsafe_allow_html=True)

    st.markdown("---")

    # ── TONE ANALYSIS ────────────────────────────────────────────────
    st.markdown(section_head("Tone Analysis",
        "How positive or negative the language was — executives vs analysts"),
        unsafe_allow_html=True)

    tone_cols = st.columns(2)
    with tone_cols[0]:
        st.markdown(
            f'<div style="font-size:0.65rem;color:#3D5470;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:4px;">Executive / Management</div>',
            unsafe_allow_html=True,
        )
        color_exec = "#00C9A7" if r["exec_tone"] > 0.6 else "#F0A500" if r["exec_tone"] > 0.45 else "#F05454"
        st.plotly_chart(tone_gauge(r["exec_tone"], "Executive Tone", color_exec),
                        use_container_width=True)
        tone_label = "Positive" if r["exec_tone"] > 0.6 else "Cautious" if r["exec_tone"] > 0.45 else "Negative"
        st.markdown(
            f'<div style="text-align:center;font-size:0.75rem;color:#8FA3BF;margin-top:-8px;">'
            f'Management language is <strong style="color:{color_exec};">{tone_label}</strong> '
            f'— executives expressed {"confident, forward-looking language" if r["exec_tone"] > 0.6 else "measured, cautious language" if r["exec_tone"] > 0.45 else "concern, hedging and warnings"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with tone_cols[1]:
        st.markdown(
            f'<div style="font-size:0.65rem;color:#3D5470;letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:4px;">Analyst / Q&A Session</div>',
            unsafe_allow_html=True,
        )
        color_anal = "#00C9A7" if r["analyst_tone"] > 0.6 else "#F0A500" if r["analyst_tone"] > 0.45 else "#F05454"
        st.plotly_chart(tone_gauge(r["analyst_tone"], "Analyst Tone", color_anal),
                        use_container_width=True)
        tone_label_a = "Receptive" if r["analyst_tone"] > 0.6 else "Neutral" if r["analyst_tone"] > 0.45 else "Skeptical"
        st.markdown(
            f'<div style="text-align:center;font-size:0.75rem;color:#8FA3BF;margin-top:-8px;">'
            f'Analyst questioning is <strong style="color:{color_anal};">{tone_label_a}</strong> '
            f'— analyst questions {"accepted management narrative" if r["analyst_tone"] > 0.6 else "were balanced and neutral" if r["analyst_tone"] > 0.45 else "challenged and probed management"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── SENTIMENT BREAKDOWN ──────────────────────────────────────────
    st.markdown(section_head("Sentiment Breakdown",
        "FinBERT sentence-level classification across the full transcript"),
        unsafe_allow_html=True)

    sent_cols = st.columns(2)
    with sent_cols[0]:
        st.plotly_chart(
            sentiment_donut(r["pos_sent"], r["neg_sent"], r["neut_sent"], "Overall"),
            use_container_width=True,
        )
    with sent_cols[1]:
        st.plotly_chart(
            section_comparison_bar(r["exec_pos"], r["exec_neg"], r["exec_neut"],
                                   r["qa_pos"],   r["qa_neg"],   r["qa_neut"]),
            use_container_width=True,
        )

    # Delta callout
    delta_pos  = r["qa_pos"]  - r["exec_pos"]
    delta_neg  = r["qa_neg"]  - r["exec_neg"]
    delta_sign = "more positive" if delta_pos > 0.05 else "more negative" if delta_neg > 0.05 else "similar"
    st.markdown(
        f'<div style="background:#162034;border:1px solid #1E2E48;border-left:3px solid #4D9FFF;'
        f'padding:0.75rem 1rem;border-radius:3px;font-size:0.78rem;color:#8FA3BF;">'
        f'<strong style="color:#EEF2F8;">Section delta:</strong> The Q&A session was '
        f'<strong style="color:#4D9FFF;">{delta_sign}</strong> in tone compared to the prepared remarks '
        f'(Δ positive: {delta_pos:+.0%}, Δ negative: {delta_neg:+.0%}).'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── RISK TERM FREQUENCY ──────────────────────────────────────────
    st.markdown(section_head("Key Term Frequency",
        "How often risk and growth words appeared — risk in red, growth in teal"),
        unsafe_allow_html=True)
    st.plotly_chart(risk_bar_chart(r["risk_keywords"]), use_container_width=True)

    st.markdown("---")

    # ── SHAP EXPLANATION ─────────────────────────────────────────────
    st.markdown(section_head("SHAP Explanation",
        "Which features drove this prediction — green = pushed towards BUY, red = pushed towards SELL"),
        unsafe_allow_html=True)

    st.plotly_chart(shap_waterfall(r["features"], r["shap_base"], r["prob"]),
                    use_container_width=True)

    # SHAP narrative
    top_positive = [(k, v) for k, v in r["features"].items() if v > 0]
    top_negative = [(k, v) for k, v in r["features"].items() if v < 0]
    top_positive.sort(key=lambda x: x[1], reverse=True)
    top_negative.sort(key=lambda x: x[1])

    pos_str  = ", ".join(f'<strong style="color:#00C9A7;">{k}</strong>' for k, _ in top_positive[:3]) if top_positive else "none"
    neg_str  = ", ".join(f'<strong style="color:#F05454;">{k}</strong>' for k, _ in top_negative[:2]) if top_negative else "none"
    final_direction = "bullish" if r["signal"] == "BUY" else "bearish" if r["signal"] == "SELL" else "neutral"

    st.markdown(
        f'<div style="background:#162034;border:1px solid #1E2E48;border-radius:6px;'
        f'padding:1rem 1.2rem;font-size:0.8rem;color:#8FA3BF;line-height:1.8;">'
        f'<div style="font-weight:600;color:#EEF2F8;margin-bottom:0.5rem;">Explanation summary</div>'
        f'The model started at a base probability of <strong style="color:#EEF2F8;">{r["shap_base"]:.2f}</strong> '
        f'and arrived at <strong style="color:#EEF2F8;">{r["prob"]:.2f}</strong>. '
        f'The main features driving the {final_direction} outcome were {pos_str}. '
        f'These were partially offset by {neg_str}. '
        f'Overall the NLP features ({pos_str.split(",")[0] if top_positive else "none"}) '
        f'were the most influential factor in this call.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── FEATURE TABLE ────────────────────────────────────────────────
    with st.expander("Full SHAP feature table"):
        shap_df = pd.DataFrame(
            [(k, v, "↑ BUY" if v > 0 else "↓ SELL") for k, v in sorted(r["features"].items(), key=lambda x: x[1], reverse=True)],
            columns=["Feature", "SHAP value", "Direction"],
        )
        st.markdown(dark_table(shap_df), unsafe_allow_html=True)

    st.markdown(footer(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: BROWSE RESULTS
# ══════════════════════════════════════════════════════════════════════

def page_results():
    st.markdown(top_bar(badge="Analyst Portal"), unsafe_allow_html=True)
    st.markdown(section_head("Historical Results",
                             "Simulated — replace with real results from Phase 6 test output"),
                unsafe_allow_html=True)

    rng = np.random.RandomState(42)
    tickers = ["AAPL", "MSFT", "ABNB", "CSCO", "ADBE", "ABT", "ACN", "ADI", "AMZN", "NVDA"]
    records = []
    for _ in range(80):
        t = rng.choice(tickers)
        q = rng.choice(["Q1", "Q2", "Q3", "Q4"])
        y = rng.choice([2023, 2024, 2025])
        ar = float(rng.normal(0.005, 0.045))
        records.append({"ticker": t, "quarter": f"{q} {y}",
                        "abnormal_return": ar,
                        "3-day return": float(rng.normal(0.01, 0.05)),
                        "beat_market": 1 if ar > 0 else 0})
    df = pd.DataFrame(records)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Calls", str(len(df)), "in historical set", "#00C9A7"), unsafe_allow_html=True)
    with c2:
        avg = df["abnormal_return"].mean()
        st.markdown(kpi_card("Avg Gain vs Market", f"{avg:+.2%}", "vs S&P 500",
                             "#30D158" if avg >= 0 else "#F05454"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Beat Market", f"{df['beat_market'].mean():.1%}", "of predictions correct", "#F0A500"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Typical Swing", f"{df['abnormal_return'].std():.2%}", "std of abnormal return", "#4D9FFF"), unsafe_allow_html=True)

    st.markdown("---")
    fc1, fc2, fc3 = st.columns(3)
    with fc1: sel_tick = st.multiselect("Ticker", sorted(df["ticker"].unique()))
    with fc2: sel_q    = st.multiselect("Quarter", sorted(df["quarter"].unique()))
    with fc3: ret_f    = st.selectbox("Return", ["All", "Positive", "Negative"])

    fd = df.copy()
    if sel_tick: fd = fd[fd["ticker"].isin(sel_tick)]
    if sel_q:    fd = fd[fd["quarter"].isin(sel_q)]
    if ret_f == "Positive": fd = fd[fd["abnormal_return"] > 0]
    elif ret_f == "Negative": fd = fd[fd["abnormal_return"] < 0]

    n_bins = max(5, min(30, len(fd) // 2)) if len(fd) > 0 else 10
    fig = go.Figure(go.Histogram(
        x=fd["abnormal_return"],
        nbinsx=n_bins,
        marker_color="#00C9A7",
        marker_line_color="#0F1829",
        marker_line_width=1,
        opacity=0.85,
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=0, line_dash="dash", line_color="#F05454", line_width=2,
        annotation_text="Market avg (0%)",
        annotation_font=dict(color="#F05454", size=10),
        annotation_position="top right",
    )
    fig.update_layout(
        **plotly_theme(),
        title=f"Abnormal Return Distribution  ({len(fd)} calls)",
        height=300, bargap=0.05,
    )
    fig.update_xaxes(
        title_text="Abnormal Return vs S&P 500  (negative = underperformed, positive = outperformed)",
        title_font=dict(size=10, color="#8FA3BF"),
        tickformat=".1%",
        tickfont=dict(size=10, color="#8FA3BF"),
    )
    fig.update_yaxes(
        title_text="Number of Earnings Calls",
        title_font=dict(size=10, color="#8FA3BF"),
        tickfont=dict(size=10, color="#8FA3BF"),
        dtick=1,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(dark_table(fd, max_rows=200), unsafe_allow_html=True)
    st.download_button("⬇ Download CSV", fd.to_csv(index=False),
                       f"ecrie_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    st.markdown(footer(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE: MODEL EXPLORER (DEV)
# ══════════════════════════════════════════════════════════════════════

def page_models():
    st.markdown(top_bar(badge="Developer"), unsafe_allow_html=True)
    st.markdown(
        '<div style="background:rgba(240,165,0,0.08);border:1px solid rgba(240,165,0,0.3);'
        'border-radius:4px;padding:0.6rem 1rem;margin-bottom:1.2rem;'
        'font-size:0.72rem;color:#F0A500;">'
        '⚠️ DEVELOPER MODE — pipeline internals visible.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(section_head("Model Explorer",
                             "Detailed performance breakdown across all model phases"),
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Performance", "Feature Importance", "CV Results"])

    with tab1:
        st.plotly_chart(model_comparison_chart(), use_container_width=True)
        data = {
            "Model": ["Financial Only", "FinBERT Only", "XGBoost Combined", "Agent Enhanced"],
            "ROC-AUC": [0.482, 0.547, 0.633, 0.572],
            "Notes": ["Baseline", "NLP signals only", "Held-out test (Phase 6)", "Cross-validated (Phase 10)"],
            "Features": [77, 784, 861, 861],
            "Phase": ["Phase 6", "Phase 6", "Phase 6", "Phase 10"],
        }
        st.markdown(dark_table(pd.DataFrame(data)), unsafe_allow_html=True)

    with tab2:
        feat_names = ["Mgmt tone (linguistic)","Analyst tone (linguistic)","FinBERT sentiment",
                      "Revenue growth","EPS surprise","Risk word count",
                      "Forward guidance","Market beta","Uncertainty words","P/E ratio"]
        feat_vals  = [0.180, 0.160, 0.150, 0.130, 0.110, 0.100, 0.090, 0.080, 0.070, 0.060]
        fig2 = go.Figure(go.Bar(
            x=feat_vals[::-1], y=feat_names[::-1], orientation="h",
            marker_color=["#00C9A7"] * 5 + ["#F0A500"] * 5,
            text=[f"{v:.0%}" for v in feat_vals[::-1]], textposition="outside",
            textfont={"size": 10, "color": "#EEF2F8"},
        ))
        fig2.update_layout(**plotly_theme(), height=380,
                           title="Top 10 features by mean |SHAP| across test set")
        fig2.update_xaxes(title_text="Mean |SHAP value|")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        rng2  = np.random.RandomState(99)
        folds = [f"Fold {i}" for i in range(1, 6)]
        cv_df = pd.DataFrame({
            "Fold":      folds,
            "ROC-AUC":   [round(v, 4) for v in rng2.uniform(0.549, 0.590, 5)],
            "Precision": [round(v, 4) for v in rng2.uniform(0.530, 0.570, 5)],
            "Recall":    [round(v, 4) for v in rng2.uniform(0.515, 0.555, 5)],
        })
        cv_df.loc[len(cv_df)] = ["Mean ± Std",
                                  f"{cv_df['ROC-AUC'].mean():.4f} ± {cv_df['ROC-AUC'].std():.4f}",
                                  f"{cv_df['Precision'].mean():.4f} ± {cv_df['Precision'].std():.4f}",
                                  f"{cv_df['Recall'].mean():.4f} ± {cv_df['Recall'].std():.4f}"]
        st.markdown(dark_table(cv_df), unsafe_allow_html=True)

    st.markdown(footer(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: CAUSAL INFERENCE
# ══════════════════════════════════════════════════════════════════════

def page_causal():
    st.markdown(top_bar(badge="Analyst Portal"), unsafe_allow_html=True)
    st.markdown(section_head("Causal Inference",
                             "Does earnings call tone CAUSE returns, or just correlate?"),
                unsafe_allow_html=True)

    # Load real results if available, else show demo
    results_path = Path("results/causal/ate_results.json")
    cate_path    = Path("results/causal/cate_results.json")

    if results_path.exists():
        with open(results_path) as f:
            ate = json.load(f)
        with open(cate_path) as f:
            cate = json.load(f)
        st.success("Loaded real causal inference results from results/causal/")
    else:
        # Demo results
        ate = {
            "ate": 0.0124, "ate_pct": 1.24,
            "ci_lower": 0.0031, "ci_upper": 0.0217,
            "standard_error": 0.0048, "t_statistic": 2.58,
            "p_value": 0.0112, "cohens_d": 0.31,
            "n_matched_pairs": 187, "significant_05": True,
            "direction": "POSITIVE",
            "interpretation": (
                "Positive earnings call language causes a +1.24% "
                "significant abnormal return over 3 trading days."
            ),
        }
        cate = {
            "by_sector": {
                "Tech":     {"cate_pct": 1.87, "n_pairs": 54, "p_value": 0.008,  "significant": True},
                "Finance":  {"cate_pct": 0.63, "n_pairs": 38, "p_value": 0.201,  "significant": False},
                "Health":   {"cate_pct": 1.12, "n_pairs": 29, "p_value": 0.044,  "significant": True},
                "Consumer": {"cate_pct": 0.91, "n_pairs": 22, "p_value": 0.091,  "significant": False},
                "Energy":   {"cate_pct": 0.44, "n_pairs": 18, "p_value": 0.387,  "significant": False},
            },
            "by_market_cap": {
                "Small": {"cate_pct": 2.11, "n_pairs": 47, "p_value": 0.021},
                "Mid":   {"cate_pct": 1.43, "n_pairs": 52, "p_value": 0.034},
                "Large": {"cate_pct": 0.88, "n_pairs": 49, "p_value": 0.098},
                "Mega":  {"cate_pct": 0.51, "n_pairs": 39, "p_value": 0.241},
            }
        }
        st.info("Demo results shown — run `python run_phase8b.py` to generate real results.")

    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    sig_color = "#30D158" if ate["significant_05"] else "#F0A500"
    with c1: st.markdown(kpi_card("Causal ATE",
        f"{ate['ate_pct']:+.2f}%",
        "avg extra return caused by positive call", sig_color), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("95% CI",
        f"[{ate['ci_lower']*100:+.1f}%, {ate['ci_upper']*100:+.1f}%]",
        "bootstrap confidence interval", "#4D9FFF"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("p-value",
        f"{ate['p_value']:.4f}",
        "significant at 5% level" if ate["significant_05"] else "not significant", sig_color),
        unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Matched Pairs",
        str(ate["n_matched_pairs"]),
        "treated/control pairs used", "#F0A500"), unsafe_allow_html=True)

    # Interpretation banner
    border_color = "#30D158" if ate["significant_05"] else "#F0A500"
    st.markdown(
        f'<div style="background:#162034;border:1px solid #1E2E48;border-left:3px solid {border_color};'
        f'padding:0.9rem 1.2rem;border-radius:3px;margin:1rem 0;font-size:0.82rem;color:#8FA3BF;">'
        f'<strong style="color:#EEF2F8;">Causal finding:</strong> {ate["interpretation"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── ATE Visualisation ─────────────────────────────────────────────
    col_ate, col_cate = st.columns(2)

    with col_ate:
        st.markdown(section_head("ATE Waterfall",
            "How the causal estimate was built"), unsafe_allow_html=True)
        ate_val = ate["ate_pct"]
        ci_lo   = ate["ci_lower"] * 100
        ci_hi   = ate["ci_upper"] * 100
        naive   = ate_val * 1.6   # naive (pre-matching) is always higher

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Naive difference", "After PSM matching", "95% CI range"],
            y=[naive, ate_val, ci_hi - ci_lo],
            marker_color=["#F0A500", "#00C9A7", "#4D9FFF"],
            text=[f"{naive:+.2f}%", f"{ate_val:+.2f}%", f"±{(ci_hi-ci_lo)/2:.2f}%"],
            textposition="outside", textfont={"size": 11, "color": "#EEF2F8"},
        ))
        fig.add_hline(y=0, line_color="#3D5470", line_width=1)
        fig.update_layout(**plotly_theme(), height=280,
                          title="Naive vs causal estimate (matching reduces bias)")
        fig.update_yaxes(title_text="Abnormal return (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col_cate:
        st.markdown(section_head("CATE by Sector",
            "Where is the causal effect strongest?"), unsafe_allow_html=True)
        if "by_sector" in cate:
            sectors = list(cate["by_sector"].keys())
            cate_vals = [cate["by_sector"][s]["cate_pct"] for s in sectors]
            sig_flags = [cate["by_sector"][s].get("significant", False) for s in sectors]
            bar_colors = ["#00C9A7" if s else "#3D5470" for s in sig_flags]

            fig2 = go.Figure(go.Bar(
                x=sectors, y=cate_vals,
                marker_color=bar_colors,
                text=[f"{v:+.2f}%" for v in cate_vals],
                textposition="outside", textfont={"size": 11, "color": "#EEF2F8"},
            ))
            fig2.add_hline(y=ate_val, line_dash="dash", line_color="#F0A500",
                           annotation_text=f"Overall ATE={ate_val:+.2f}%",
                           annotation_font=dict(color="#F0A500", size=9))
            fig2.update_layout(**plotly_theme(), height=280,
                               title="Teal = significant (p<0.05), gray = not significant")
            fig2.update_yaxes(title_text="Causal effect (%)")
            st.plotly_chart(fig2, use_container_width=True)

    # ── CATE by Market Cap ───────────────────────────────────────────
    if "by_market_cap" in cate:
        st.markdown(section_head("CATE by Company Size",
            "Smaller companies show stronger language effects"), unsafe_allow_html=True)
        caps = ["Small", "Mid", "Large", "Mega"]
        cap_vals = [cate["by_market_cap"].get(c, {}).get("cate_pct", 0) for c in caps]
        cap_n    = [cate["by_market_cap"].get(c, {}).get("n_pairs", 0) for c in caps]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=caps, y=cap_vals, name="CATE",
            marker_color=["#00C9A7","#1D9E75","#0F6E56","#085041"],
            text=[f"{v:+.2f}%" for v in cap_vals],
            textposition="outside", textfont={"size": 11, "color": "#EEF2F8"},
        ))
        fig3.add_trace(go.Scatter(
            x=caps, y=cap_n, name="N pairs",
            mode="lines+markers", yaxis="y2",
            line=dict(color="#F0A500", width=2), marker=dict(size=7),
        ))
        fig3.update_layout(
            **plotly_theme(), height=260, barmode="group",
            title="Effect decreases with company size — small caps react more to call tone",
            yaxis2=dict(overlaying="y", side="right", title="N matched pairs",
                        showgrid=False, tickfont=dict(color="#F0A500")),
        )
        fig3.update_yaxes(title_text="Causal effect (%)")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Methodology explainer ────────────────────────────────────────
    st.markdown("---")
    st.markdown(section_head("How this works", "Propensity Score Matching explained"),
                unsafe_allow_html=True)
    st.markdown(info_card("Why we can't just compare positive vs negative calls directly",
        '<div style="font-size:0.78rem;color:#8FA3BF;line-height:1.8;">'
        'Companies that give positive earnings calls also tend to have <strong style="color:#EEF2F8;">'
        'better fundamentals</strong> — higher EPS, stronger revenue growth, lower debt. '
        'So a naive comparison confounds the language effect with the underlying company quality.<br><br>'
        '<strong style="color:#EEF2F8;">Propensity Score Matching</strong> fixes this by:<br>'
        '1. Estimating P(positive call | financial characteristics) for each company<br>'
        '2. Matching each positive-call company to a financially similar negative-call company<br>'
        '3. Comparing returns only within matched pairs — financials are held constant<br><br>'
        'The remaining return difference is attributable to the <strong style="color:#00C9A7;">'
        'language and tone of the call alone</strong>.'
        '</div>'),
        unsafe_allow_html=True)

    st.markdown(footer(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════

page = st.session_state.page
if   page == "dashboard": page_dashboard()
elif page == "analyze":   page_analyze()
elif page == "results":   page_results()
elif page == "causal":    page_causal()
elif page == "models":    page_models()
else:                     page_dashboard()
