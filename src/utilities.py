"""Helpers: risk labels, alerts, simulation, Plotly charts, UI styling."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    ALERT_THRESHOLD,
    METADATA_PATH,
    RISK_BANDS,
    TARGET_COLUMN,
)


def load_metadata() -> dict | None:
    """Load training metadata JSON if present."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def probability_to_risk_level(probability: float) -> str:
    """Map fraud probability to Low / Medium / High / Critical."""
    for bound, label in RISK_BANDS:
        if probability < bound:
            return label
    return "Critical"


def risk_level_rank(level: str) -> int:
    """Numeric rank for filtering (higher = riskier)."""
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    return order.get(level, 0)


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def risk_color(level: str) -> str:
    """Hex colors for fintech dashboard."""
    return {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#f97316",
        "Critical": "#ef4444",
    }.get(level, "#6b7280")


def apply_risk_columns(
    df: pd.DataFrame,
    probabilities: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Add scoring columns to a copy of the dataframe."""
    out = df.copy()
    out["fraud_probability"] = probabilities.values
    out["risk_level"] = out["fraud_probability"].apply(probability_to_risk_level)
    out["is_flagged"] = out["fraud_probability"] >= threshold
    out["alert"] = out["fraud_probability"] >= ALERT_THRESHOLD
    out["predicted_fraud"] = (out["fraud_probability"] >= threshold).astype(int)
    return out


def filter_suspicious(df: pd.DataFrame, min_risk: str = "Medium") -> pd.DataFrame:
    """Keep rows at or above min_risk tier."""
    min_rank = risk_level_rank(min_risk)
    mask = df["risk_level"].map(risk_level_rank) >= min_rank
    return df.loc[mask].sort_values("fraud_probability", ascending=False)


def check_alerts(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that triggered alert threshold."""
    if "alert" not in scored_df.columns:
        return scored_df.iloc[0:0]
    return scored_df.loc[scored_df["alert"]].copy()


def generate_simulated_transaction(
    rng: np.random.Generator,
    historical_df: pd.DataFrame,
) -> pd.Series:
    """
    Bootstrap a row from historical data and add small Gaussian noise
    to Amount and V-features for variety.
    """
    row = historical_df.sample(n=1, random_state=int(rng.integers(0, 2**31))).iloc[0].copy()
    row["Time"] = float(row["Time"]) + float(rng.uniform(0, 3600))
    row["Amount"] = max(0.01, float(row["Amount"]) * float(rng.lognormal(0, 0.15)))
    for col in [c for c in historical_df.columns if c.startswith("V")]:
        row[col] = float(row[col]) + float(rng.normal(0, 0.05))
    if TARGET_COLUMN in row.index:
        row = row.drop(labels=[TARGET_COLUMN])
    return row


def is_dark_theme() -> bool:
    """Detect Streamlit dark mode (runtime theme toggle)."""
    try:
        import streamlit as st

        theme = getattr(st.context, "theme", None) if hasattr(st, "context") else None
        if theme is not None and getattr(theme, "base", None):
            return theme.base == "dark"
        base = st.get_option("theme.base")
        if base:
            return str(base).lower() == "dark"
    except Exception:
        pass
    return False


def get_plotly_theme(dark: bool | None = None) -> dict[str, str]:
    """Colors for Plotly charts that match Streamlit light/dark UI."""
    if dark is None:
        dark = is_dark_theme()
    if dark:
        return {
            "font_color": "#e2e8f0",
            "title_color": "#f8fafc",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "grid_color": "#475569",
            "legit_color": "#38bdf8",
        }
    return {
        "font_color": "#0f172a",
        "title_color": "#0f172a",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "grid_color": "#cbd5e1",
        "legit_color": "#1e3a5f",
    }


def apply_plotly_theme(fig: go.Figure, dark: bool | None = None) -> go.Figure:
    """Apply readable fonts, axes, and grids for the active Streamlit theme."""
    theme = get_plotly_theme(dark)
    fig.update_layout(
        font=dict(color=theme["font_color"], family="Segoe UI, sans-serif"),
        title_font_color=theme["title_color"],
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        legend=dict(font=dict(color=theme["font_color"])),
        coloraxis_colorbar=dict(tickfont=dict(color=theme["font_color"])),
    )
    fig.update_xaxes(
        color=theme["font_color"],
        title_font_color=theme["font_color"],
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["grid_color"],
    )
    fig.update_yaxes(
        color=theme["font_color"],
        title_font_color=theme["font_color"],
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["grid_color"],
    )
    fig.update_traces(textfont=dict(color=theme["font_color"]))
    return fig


def _chart_layout_kwargs(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15)),
        margin=dict(l=24, r=24, t=56, b=24),
        height=380,
    )


def plot_fraud_pie(fraud_count: int, legit_count: int) -> go.Figure:
    theme = get_plotly_theme()
    fig = px.pie(
        names=["Legitimate", "Fraud"],
        values=[legit_count, fraud_count],
        color=["Legitimate", "Fraud"],
        color_discrete_map={"Legitimate": theme["legit_color"], "Fraud": "#ef4444"},
        hole=0.5,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color=theme["font_color"], width=1)),
    )
    fig.update_layout(showlegend=True, **_chart_layout_kwargs("Fraud vs legitimate mix"))
    return apply_plotly_theme(fig)


def plot_amount_distribution(
    df: pd.DataFrame,
    prob_col: str = "fraud_probability",
    amount_col: str = "Amount",
) -> go.Figure:
    """Histogram of transaction amounts colored by predicted risk."""
    plot_df = df.copy()
    plot_df["risk_bucket"] = plot_df[prob_col].apply(
        lambda p: "Fraud-like" if p >= 0.6 else "Normal-like"
    )
    fig = px.histogram(
        plot_df,
        x=amount_col,
        color="risk_bucket",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"Fraud-like": "#ef4444", "Normal-like": "#0ea5e9"},
    )
    fig.update_layout(
        xaxis_title="Amount ($)",
        yaxis_title="Count",
        bargap=0.12,
        **_chart_layout_kwargs("Transaction amount distribution"),
    )
    return apply_plotly_theme(fig)


def plot_fraud_trend(
    df: pd.DataFrame,
    time_col: str = "Time",
    flag_col: str = "is_flagged",
    n_bins: int = 30,
) -> go.Figure:
    """Fraud flag rate over binned time."""
    if time_col not in df.columns or df.empty:
        fig = go.Figure()
        theme = get_plotly_theme()
        fig.add_annotation(
            text="No time data",
            showarrow=False,
            font=dict(color=theme["font_color"]),
        )
        return apply_plotly_theme(fig)

    plot_df = df.copy()
    plot_df["time_bin"] = pd.qcut(plot_df[time_col], q=min(n_bins, len(plot_df)), duplicates="drop")
    agg = plot_df.groupby("time_bin", observed=True).agg(
        flag_rate=(flag_col, "mean"),
        count=(flag_col, "count"),
    ).reset_index()
    agg["bin_label"] = range(len(agg))

    fig = px.line(
        agg,
        x="bin_label",
        y="flag_rate",
        markers=True,
        labels={"bin_label": "Time period", "flag_rate": "Flagged rate"},
    )
    fig.update_traces(
        line=dict(color="#0ea5e9", width=2.5),
        marker=dict(size=7, line=dict(width=1, color="#0ea5e9")),
        fill="tozeroy",
        fillcolor="rgba(14, 165, 233, 0.12)",
    )
    fig.update_layout(**_chart_layout_kwargs("Flagged transaction rate over time"))
    return apply_plotly_theme(fig)


def plot_risk_distribution(scored_df: pd.DataFrame) -> go.Figure:
    counts = scored_df["risk_level"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"], fill_value=0
    )
    colors = [risk_color(l) for l in counts.index]
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=colors,
            marker_line=dict(width=0),
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Risk tier",
        yaxis_title="Transactions",
        **_chart_layout_kwargs("Risk level distribution"),
    )
    return apply_plotly_theme(fig)


def style_risk_dataframe(df: pd.DataFrame) -> pd.DataFrame | Any:
    """Highlight risk_level and fraud_probability for Streamlit display."""
    styler = df.style

    if "risk_level" in df.columns:

        def _risk_bg(val: str) -> str:
            c = risk_color(str(val))
            return f"background-color: {c}22; color: {c}; font-weight: 600;"

        styler = styler.map(_risk_bg, subset=["risk_level"])

    if "fraud_probability" in df.columns:

        def _prob_style(val) -> str:
            try:
                p = float(val)
            except (TypeError, ValueError):
                return ""
            if p >= 0.85:
                return "color: #ef4444; font-weight: 600;"
            if p >= 0.6:
                return "color: #f97316; font-weight: 500;"
            return "color: inherit;"

        styler = styler.map(_prob_style, subset=["fraud_probability"])
        styler = styler.format({"fraud_probability": "{:.2%}"})

    if "Amount" in df.columns:
        styler = styler.format(
            {"Amount": lambda v: format_currency(v) if isinstance(v, (int, float)) else v}
        )

    return styler


def get_custom_css() -> str:
    """Fintech-style overrides; uses Streamlit theme variables for light/dark."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
    }

    /* Hero */
    .hero-block {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(30, 58, 95, 0.06) 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.5rem 1rem;
        margin-bottom: 1.25rem;
    }
    .main-header {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-color) !important;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .sub-header {
        color: var(--text-color) !important;
        opacity: 0.72;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.45;
    }

    /* Page sections */
    .page-header h2 {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 0 0 0.2rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0ea5e9;
        display: inline-block;
        min-width: 120px;
    }
    .page-header p {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.88rem;
        margin: 0.5rem 0 1rem 0;
    }

    /* Sidebar brand */
    [data-testid="stSidebar"] .sidebar-brand {
        background: linear-gradient(160deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 10px;
        padding: 1rem 1rem 0.85rem;
        margin-bottom: 1rem;
        text-align: left;
    }
    [data-testid="stSidebar"] .sidebar-brand-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] .sidebar-brand-sub {
        color: #94a3b8;
        font-size: 0.78rem;
        margin: 0.2rem 0 0.65rem 0;
    }
    [data-testid="stSidebar"] .status-pill {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
    }
    [data-testid="stSidebar"] .status-ok {
        background: rgba(16, 185, 129, 0.25);
        color: #6ee7b7;
    }
    [data-testid="stSidebar"] .status-warn {
        background: rgba(245, 158, 11, 0.25);
        color: #fcd34d;
    }

    /* KPI row — accent borders per column */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] {
        border-left: 4px solid #0ea5e9;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {
        border-left: 4px solid #8b5cf6;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
        border-left: 4px solid #f59e0b;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(4) div[data-testid="stMetric"] {
        border-left: 4px solid #ef4444;
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.12);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] p {
        color: var(--text-color) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        opacity: 0.85;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-weight: 500;
    }

    /* Chart panels */
    .chart-panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 0.75rem 0.5rem 0.25rem;
        margin-bottom: 0.5rem;
    }

    /* Model performance strip */
    .model-strip {
        background: var(--secondary-background-color);
        border-radius: 10px;
        border: 1px solid rgba(14, 165, 233, 0.25);
        padding: 0.85rem 1.1rem;
        margin: 0.75rem 0 1rem 0;
        font-size: 0.9rem;
        color: var(--text-color);
    }
    .model-strip strong { color: #0ea5e9; }

    /* Control panel (live monitor) */
    .control-panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 0.5rem 0.75rem 0.25rem;
        margin-bottom: 1rem;
    }

    /* Alert banner */
    .alert-banner {
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0 12px 0;
        font-size: 0.92rem;
    }
    [data-theme="light"] .alert-banner {
        background: linear-gradient(90deg, #fef2f2 0%, #fff7ed 100%);
        color: #7f1d1d;
    }
    [data-theme="dark"] .alert-banner {
        background: linear-gradient(90deg, #450a0a 0%, #431407 100%);
        color: #fecaca;
    }

    /* Sidebar nav spacing */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.45rem 0.5rem;
        border-radius: 6px;
    }
    [data-theme="dark"] .js-plotly-plot .plotly .modebar-btn {
        color: #e2e8f0;
    }

    /* Hide Streamlit footer for cleaner demo */
    footer[data-testid="stFooter"] {
        opacity: 0.5;
    }
    </style>
    """


def render_hero(title: str, subtitle: str) -> None:
    import streamlit as st

    st.markdown(
        f'<div class="hero-block">'
        f'<p class="main-header">{title}</p>'
        f'<p class="sub-header">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "") -> None:
    import streamlit as st

    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)


def render_sidebar_brand(model_ready: bool, meta: dict | None = None) -> None:
    import streamlit as st

    status_cls = "status-ok" if model_ready else "status-warn"
    status_txt = "Model online" if model_ready else "Train required"
    pr_auc = ""
    if meta and meta.get("metrics"):
        pr_auc = f'<div style="color:#94a3b8;font-size:0.72rem;margin-top:0.5rem;">PR-AUC {meta["metrics"].get("pr_auc", 0):.3f}</div>'

    st.sidebar.markdown(
        f'<div class="sidebar-brand">'
        f'<p class="sidebar-brand-title">FraudGuard</p>'
        f'<p class="sidebar-brand-sub">Transaction monitoring</p>'
        f'<span class="status-pill {status_cls}">{status_txt}</span>'
        f"{pr_auc}</div>",
        unsafe_allow_html=True,
    )


def render_model_strip(meta: dict) -> None:
    import streamlit as st

    m = meta.get("metrics", {})
    st.markdown(
        f'<div class="model-strip">'
        f"<strong>Model</strong> PR-AUC {m.get('pr_auc', 0):.4f} &nbsp;|&nbsp; "
        f"Threshold <strong>{meta.get('threshold', 0.5):.2f}</strong> &nbsp;|&nbsp; "
        f"Trained {meta.get('trained_at', 'N/A')[:10]}"
        f"</div>",
        unsafe_allow_html=True,
    )


def banking_context_markdown() -> str:
    """Short educational copy for About expander."""
    return """
### Why fraud detection matters
Financial institutions face card-not-present fraud, account takeover, and money laundering.
Fast detection limits losses, meets regulatory expectations (BSA/AML), and protects customers.

### How banks use machine learning
Rule engines (velocity, geography, merchant category) run alongside ML models that score
each transaction in milliseconds. Anonymized features (like V1–V28 in this dataset) feed
models that are retrained as fraud patterns drift. High scores route to analysts or step-up auth.

### Fraud prevention vs customer experience
Stricter thresholds catch more fraud but increase false declines—blocked cards, support calls,
and abandoned purchases. Banks tune thresholds by segment, channel, and loss appetite.

### False positives
A false alert wastes analyst time and frustrates customers. With ~0.17% fraud prevalence,
**precision–recall tradeoffs** matter more than accuracy. This project optimizes PR-AUC and
uses configurable thresholds to reflect that balance.
    """

