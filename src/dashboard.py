"""Streamlit dashboard: fraud monitoring UI."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    SIM_MAX_QUEUE,
    TARGET_COLUMN,
)
from src.prediction import (
    get_inference_threshold,
    model_exists,
    score_uploaded_csv,
)
from src.preprocessing import load_data, validate_upload_schema
from src.utilities import (
    apply_plotly_theme,
    banking_context_markdown,
    check_alerts,
    filter_suspicious,
    format_currency,
    generate_simulated_transaction,
    get_custom_css,
    is_dark_theme,
    load_metadata,
    plot_amount_distribution,
    plot_fraud_pie,
    plot_fraud_trend,
    plot_risk_distribution,
    render_hero,
    render_model_strip,
    render_page_header,
    render_sidebar_brand,
    style_risk_dataframe,
)


@st.cache_resource
def _cached_model():
    from src.prediction import load_model

    return load_model()


@st.cache_data
def _cached_sample_data(max_rows: int = 5000):
    if not DATA_PATH.exists():
        return None
    df = load_data(DATA_PATH)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    return df


def _init_session_state():
    defaults = {
        "live_queue": pd.DataFrame(),
        "live_total": 0,
        "live_flagged": 0,
        "live_alerts": 0,
        "last_alert_ids": set(),
        "sim_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _score_with_cache(df: pd.DataFrame) -> pd.DataFrame:
    model = _cached_model()
    threshold = get_inference_threshold()
    from src.prediction import predict_proba
    from src.utilities import apply_risk_columns

    proba = predict_proba(df, model=model)
    return apply_risk_columns(df, proba, threshold)


def _plot_chart(fig):
    """Render Plotly figure inside a bordered panel."""
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)


def _kpi_row(scored: pd.DataFrame):
    total = len(scored)
    flagged = int(scored["is_flagged"].sum()) if "is_flagged" in scored.columns else 0
    alerts = int(scored["alert"].sum()) if "alert" in scored.columns else 0

    fraud_count = fraud_pct = None
    if TARGET_COLUMN in scored.columns:
        fraud_count = int((scored[TARGET_COLUMN] == 1).sum())
        fraud_pct = 100.0 * fraud_count / total if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{total:,}")
    if fraud_count is not None:
        c2.metric("Confirmed fraud", f"{fraud_count:,}")
        c3.metric("Fraud rate", f"{fraud_pct:.3f}%")
    else:
        c2.metric(
            "High / critical",
            f"{int((scored['risk_level'].isin(['High', 'Critical'])).sum()):,}",
        )
        c3.metric("Avg fraud probability", f"{scored['fraud_probability'].mean():.2%}")
    c4.metric("Flagged by model", f"{flagged:,}", delta=f"{alerts} alerts" if alerts else None)


def _display_scored_table(df: pd.DataFrame, max_rows: int = 200):
    """Table with risk highlighting and formatted currency / probability."""
    show = df.head(max_rows).copy()
    st.dataframe(style_risk_dataframe(show), use_container_width=True, hide_index=True)


def page_overview(scored: pd.DataFrame):
    render_page_header(
        "Executive overview",
        "Portfolio snapshot of transaction volume, fraud exposure, and model performance.",
    )
    _kpi_row(scored)

    meta = load_metadata()
    if meta:
        render_model_strip(meta)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        if TARGET_COLUMN in scored.columns:
            fraud_n = int((scored[TARGET_COLUMN] == 1).sum())
            legit_n = len(scored) - fraud_n
            _plot_chart(plot_fraud_pie(fraud_n, legit_n))
        else:
            _plot_chart(plot_risk_distribution(scored))
    with col2:
        _plot_chart(plot_amount_distribution(scored))


def page_transactions(scored: pd.DataFrame):
    render_page_header(
        "Suspicious transaction monitor",
        "Review flagged authorizations sorted by fraud probability.",
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        min_risk = st.selectbox("Minimum risk", ["Medium", "High", "Critical"], index=0)
    with c2:
        suspicious = filter_suspicious(scored, min_risk=min_risk)
        st.caption(f"{len(suspicious):,} transactions at or above **{min_risk}** risk (of {len(scored):,})")

    if suspicious.empty:
        st.success("No transactions match the selected risk filter.")
        return

    display_cols = ["Time", "Amount", "fraud_probability", "risk_level", "is_flagged", "alert"]
    if TARGET_COLUMN in suspicious.columns:
        display_cols.append(TARGET_COLUMN)
    display_cols = [c for c in display_cols if c in suspicious.columns]

    _display_scored_table(suspicious[display_cols])


def page_analytics(scored: pd.DataFrame):
    render_page_header(
        "Risk analytics",
        "Distribution and temporal patterns in flagged activity.",
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        _plot_chart(plot_amount_distribution(scored))
    with col2:
        _plot_chart(plot_fraud_trend(scored))

    if TARGET_COLUMN in scored.columns and "predicted_fraud" in scored.columns:
        st.markdown("##### Model vs actual (labeled sample)")
        from sklearn.metrics import classification_report

        report = classification_report(
            scored[TARGET_COLUMN],
            scored["predicted_fraud"],
            target_names=["Legit", "Fraud"],
        )
        st.code(report, language=None)


def page_upload():
    render_page_header(
        "Upload & score",
        "Analyze a CSV batch with the trained fraud model and export results.",
    )

    with st.container(border=True):
        st.markdown("**Requirements:** `Time`, `Amount`, `V1`–`V28`. Optional `Class` for evaluation.")
        uploaded = st.file_uploader("Transaction file", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    ok, msg = validate_upload_schema(df)
    if not ok:
        st.error(msg)
        return

    if not model_exists():
        st.error("Train the model first: `python train_model.py`")
        return

    col_a, col_b = st.columns([1, 4])
    with col_a:
        run = st.button("Score batch", type="primary", use_container_width=True)

    if run:
        with st.spinner("Scoring transactions…"):
            scored, err = score_uploaded_csv(df)
        if err:
            st.error(err)
            return

        st.success(f"Completed — {len(scored):,} transactions scored.")
        _kpi_row(scored)

        alerts = check_alerts(scored)
        if not alerts.empty:
            st.warning(f"{len(alerts)} transaction(s) triggered high-severity alerts.")

        preview_cols = [
            c
            for c in ["Time", "Amount", "fraud_probability", "risk_level", "is_flagged", "alert", TARGET_COLUMN]
            if c in scored.columns
        ]
        _display_scored_table(scored[preview_cols], max_rows=100)

        st.download_button(
            "Download scored CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


def page_live_monitor():
    render_page_header(
        "Live monitor",
        "Simulated authorization stream with real-time risk scoring and alerts.",
    )

    sample = _cached_sample_data(10000)
    if sample is None:
        st.warning("Place `data/creditcard.csv` on disk to enable simulation.")
        return

    hist = sample.drop(columns=[TARGET_COLUMN], errors="ignore")

    with st.container(border=True):
        st.markdown("**Simulation controls**")
        c1, c2, c3, c4 = st.columns(4)
        interval = c1.slider("Interval (sec)", 0.5, 3.0, 1.0, 0.5)
        batch = c2.number_input("Batch size", 1, 5, 1)
        auto = c3.toggle("Auto-run", value=False)
        run_once = c4.button("Run batch", type="primary", use_container_width=True)

    if run_once or auto:
        rng = np.random.default_rng()
        new_rows = [
            generate_simulated_transaction(rng, hist) for _ in range(int(batch))
        ]
        batch_df = pd.DataFrame(new_rows)

        try:
            scored_batch = _score_with_cache(batch_df)
        except Exception as e:
            st.error(str(e))
            return

        queue = st.session_state.live_queue
        st.session_state.live_queue = (
            scored_batch
            if queue.empty
            else pd.concat([queue, scored_batch], ignore_index=True).tail(SIM_MAX_QUEUE).reset_index(drop=True)
        )

        st.session_state.live_total += len(scored_batch)
        st.session_state.live_flagged += int(scored_batch["is_flagged"].sum())
        st.session_state.live_alerts += int(scored_batch["alert"].sum())

        for _, row in check_alerts(scored_batch).iterrows():
            aid = (float(row["Time"]), float(row["Amount"]), float(row["fraud_probability"]))
            if aid not in st.session_state.last_alert_ids:
                st.session_state.last_alert_ids.add(aid)
                st.toast(
                    f"{row['risk_level']}: P(fraud)={row['fraud_probability']:.1%}, "
                    f"{format_currency(row['Amount'])}",
                    icon="🚨",
                )

        if auto:
            time.sleep(interval)
            st.rerun()

    queue = st.session_state.live_queue
    if queue.empty:
        st.info("Start the simulation to populate the live transaction queue.")
        return

    st.markdown("##### Session metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed", f"{st.session_state.live_total:,}")
    m2.metric("Flagged", f"{st.session_state.live_flagged:,}")
    m3.metric("Alerts", f"{st.session_state.live_alerts:,}")
    m4.metric("In queue", f"{len(queue):,}")

    recent_alerts = check_alerts(queue.tail(50))
    if not recent_alerts.empty:
        st.markdown(
            '<div class="alert-banner"><strong>Active alerts</strong> — '
            "high-risk authorizations require review.</div>",
            unsafe_allow_html=True,
        )
        _display_scored_table(
            recent_alerts[["Time", "Amount", "fraud_probability", "risk_level"]].sort_values(
                "fraud_probability", ascending=False
            ),
            max_rows=20,
        )

    st.markdown("##### Recent authorizations")
    live_cols = ["Time", "Amount", "fraud_probability", "risk_level", "is_flagged", "alert"]
    _display_scored_table(queue[live_cols].iloc[::-1].head(30), max_rows=30)

    if st.button("Clear queue", type="secondary"):
        for key in ("live_queue", "live_total", "live_flagged", "live_alerts", "last_alert_ids"):
            st.session_state[key] = pd.DataFrame() if key == "live_queue" else (set() if key == "last_alert_ids" else 0)
        st.rerun()


def page_model_info():
    render_page_header(
        "Model information",
        "Training metrics, decision threshold, and feature importance.",
    )

    meta = load_metadata()
    if not meta:
        st.warning("No training metadata found. Run `python train_model.py`.")
        return

    m = meta.get("metrics", {})
    cols = st.columns(5)
    for col, label, key in zip(
        cols,
        ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"],
        ["pr_auc", "roc_auc", "precision", "recall", "f1"],
    ):
        col.metric(label, f"{m.get(key, 0):.4f}")

    render_model_strip(meta)

    with st.expander("Raw training metadata"):
        st.json(meta)

    if not model_exists():
        return

    try:
        pipeline = _cached_model()
        clf = pipeline.named_steps["classifier"]
        names = meta.get("feature_names", [])
        if hasattr(clf, "feature_importances_") and names:
            imp = clf.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            fig = go.Figure(
                go.Bar(
                    x=[imp[i] for i in idx][::-1],
                    y=[names[i] for i in idx][::-1],
                    orientation="h",
                    marker=dict(
                        color=[imp[i] for i in idx][::-1],
                        colorscale=[[0, "#1e3a5f"], [1, "#0ea5e9"]],
                        showscale=False,
                    ),
                )
            )
            fig.update_layout(
                title=dict(text="Top 15 feature importances", x=0.02, xanchor="left"),
                height=460,
                margin=dict(l=20, r=20, t=48, b=20),
            )
            apply_plotly_theme(fig, dark=is_dark_theme())
            _plot_chart(fig)
    except Exception as e:
        st.caption(f"Could not load feature importance: {e}")


def render():
    """Main dashboard entry."""
    st.set_page_config(
        page_title="FraudGuard | Transaction Monitoring",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    ready = model_exists()
    meta = load_metadata() if ready else None
    render_sidebar_brand(ready, meta)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Transactions",
            "Analytics",
            "Upload & Score",
            "Live Monitor",
            "Model Info",
        ],
        label_visibility="collapsed",
    )

    with st.sidebar.expander("Banking context"):
        st.markdown(banking_context_markdown())

    render_hero(
        "AI Fraud Detection & Transaction Monitoring",
        "Enterprise-style risk scoring and suspicious activity monitoring — portfolio demonstration.",
    )

    if not ready:
        st.error(
            "**Model not trained.** Place `data/creditcard.csv` in the data folder, then run "
            "`python train_model.py`"
        )
        if page == "Upload & Score":
            page_upload()
        elif page == "Model Info":
            page_model_info()
        return

    scored_default = None
    if DATA_PATH.exists():
        sample = _cached_sample_data(3000)
        if sample is not None:
            try:
                scored_default = _score_with_cache(sample)
            except Exception as e:
                st.sidebar.error(f"Scoring error: {e}")

    pages = {
        "Overview": lambda: page_overview(scored_default) if scored_default is not None else st.info("Add `data/creditcard.csv` for analytics."),
        "Transactions": lambda: page_transactions(scored_default) if scored_default is not None else st.info("Add dataset for transaction views."),
        "Analytics": lambda: page_analytics(scored_default) if scored_default is not None else st.info("Add dataset for analytics."),
        "Upload & Score": page_upload,
        "Live Monitor": page_live_monitor,
        "Model Info": page_model_info,
    }
    pages[page]()
