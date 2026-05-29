import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #090b0f !important;
    color: #e8e6e1 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,255,178,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(0,180,255,0.05) 0%, transparent 50%),
        #090b0f !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 720px !important; padding: 3rem 2rem 6rem !important; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

/* ── Header block ── */
.header-block {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    animation: fadeDown 0.8s ease both;
}
.header-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #63ffb2;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    opacity: 0.85;
}
.header-title {
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #f0ede8;
    margin-bottom: 1rem;
}
.header-title span {
    background: linear-gradient(135deg, #63ffb2 0%, #00c8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: rgba(232,230,225,0.45);
    letter-spacing: 0.02em;
    line-height: 1.6;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: rgba(99,255,178,0.6);
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.6s ease both;
    animation-delay: 0.2s;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1rem;
    animation: fadeUp 0.7s ease both;
    animation-delay: 0.3s;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease;
}
.input-card:hover { border-color: rgba(99,255,178,0.15); }

/* ── Streamlit input overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: rgba(232,230,225,0.55) !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(99,255,178,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,255,178,0.06) !important;
    outline: none !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #63ffb2 0%, #00c8ff 100%) !important;
    color: #090b0f !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
    margin-top: 1.2rem !important;
    box-shadow: 0 4px 24px rgba(99,255,178,0.18) !important;
    animation: fadeUp 0.7s ease both !important;
    animation-delay: 0.5s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(99,255,178,0.28) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result cards ── */
.result-card {
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    animation: resultReveal 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    opacity: 0.06;
    background: radial-gradient(ellipse at 50% 0%, currentColor 0%, transparent 70%);
}

.result-safe {
    background: rgba(99,255,178,0.06);
    border: 1px solid rgba(99,255,178,0.25);
    color: #63ffb2;
}
.result-fraud {
    background: rgba(255,80,80,0.06);
    border: 1px solid rgba(255,80,80,0.25);
    color: #ff5858;
}

.result-status {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.8rem;
}
.result-verdict {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.6rem;
    line-height: 1.1;
}
.result-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    opacity: 0.6;
    letter-spacing: 0.05em;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    margin-top: 1.4rem;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    animation: barGrow 0.8s cubic-bezier(0.22,1,0.36,1) both;
    animation-delay: 0.3s;
}
.conf-bar-safe  { background: linear-gradient(90deg, #63ffb2, #00c8ff); }
.conf-bar-fraud { background: linear-gradient(90deg, #ff5858, #ff9a3c); }

/* ── Metrics row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 1rem;
    animation: fadeUp 0.5s ease both;
    animation-delay: 0.5s;
}
.metric-tile {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: #63ffb2;
    display: block;
    margin-bottom: 0.3rem;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(232,230,225,0.35);
}

/* ── Warning banner ── */
.warn-banner {
    background: rgba(255,154,60,0.08);
    border: 1px solid rgba(255,154,60,0.2);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: rgba(255,154,60,0.85);
    letter-spacing: 0.03em;
    line-height: 1.6;
    margin-top: 0.8rem;
    animation: fadeUp 0.5s ease both;
    animation-delay: 0.6s;
}

/* ── Footer ── */
.footer-note {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: rgba(232,230,225,0.2);
    letter-spacing: 0.08em;
    margin-top: 4rem;
    line-height: 1.8;
}

/* ── Animations ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.94) translateY(10px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes barGrow {
    from { width: 0%; }
    to   { width: var(--bar-width); }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model assets ──────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model     = joblib.load("fraud_model.pkl")
        features  = joblib.load("model_features.pkl")
        threshold = joblib.load("best_threshold.pkl")
        return model, features, float(threshold), None
    except FileNotFoundError as e:
        return None, None, 0.56, str(e)

model, feature_cols, THRESHOLD, load_error = load_assets()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <div class="header-label">PaySim — Machine Learning</div>
    <div class="header-title">Fraud <span>Detection</span><br>System</div>
    <div class="header-sub">
        AdaBoost · Threshold 0.56 · PR-AUC 0.97<br>
        Enter transaction details to classify in real time
    </div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.markdown(f"""
    <div class="warn-banner">
        Model files not found. Place fraud_model.pkl, model_features.pkl,
        and best_threshold.pkl in the same directory as app.py.<br>
        Error: {load_error}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Transaction Details</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tx_type = st.selectbox("Transaction Type",
                           options=["TRANSFER", "CASH_OUT"],
                           help="Only TRANSFER and CASH_OUT can be fraudulent in PaySim")
    amount  = st.number_input("Amount", min_value=0.0, value=50000.0, step=1000.0)
    step    = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1, step=1)

with col2:
    old_balance_orig = st.number_input("Old Balance — Origin",    min_value=0.0, value=80000.0,  step=1000.0)
    new_balance_orig = st.number_input("New Balance — Origin",    min_value=0.0, value=30000.0,  step=1000.0)
    old_balance_dest = st.number_input("Old Balance — Destination", min_value=0.0, value=0.0,   step=1000.0)
    new_balance_dest = st.number_input("New Balance — Destination", min_value=0.0, value=0.0,   step=1000.0)

st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("Analyze Transaction")


# ── Prediction ────────────────────────────────────────────────────────────────
def build_features(tx_type, amount, step,
                   old_orig, new_orig, old_dest, new_dest):
    """Build the full feature vector matching training columns."""

    # Engineered features
    amount_ratio_orig      = amount / (old_orig + 1e-9) if old_orig > 0 else 0.0
    orig_balance_zeroed    = int(new_orig == 0 and old_orig > 0)
    dest_balance_unchanged = int(old_dest == new_dest)
    balance_error_orig     = abs(old_orig - amount - new_orig)
    balance_error_dest     = abs(old_dest + amount - new_dest)

    # OHE for type
    type_CASH_OUT = int(tx_type == "CASH_OUT")
    type_DEBIT    = 0
    type_PAYMENT  = 0
    type_TRANSFER = int(tx_type == "TRANSFER")

    row = {
        "step":                   step,
        "amount":                 amount,
        "oldbalanceOrg":          old_orig,
        "newbalanceOrig":         new_orig,
        "oldbalanceDest":         old_dest,
        "newbalanceDest":         new_dest,
        "type_CASH_OUT":          type_CASH_OUT,
        "type_DEBIT":             type_DEBIT,
        "type_PAYMENT":           type_PAYMENT,
        "type_TRANSFER":          type_TRANSFER,
        "amount_ratio_orig":      amount_ratio_orig,
        "orig_balance_zeroed":    orig_balance_zeroed,
        "dest_balance_unchanged": dest_balance_unchanged,
        "balance_error_orig":     balance_error_orig,
        "balance_error_dest":     balance_error_dest,
    }

    df = pd.DataFrame([row])

    # Align to training columns
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

    return df


if predict_btn:
    with st.spinner(""):
        time.sleep(0.4)  # smooth UX delay

    input_df = build_features(
        tx_type, amount, step,
        old_balance_orig, new_balance_orig,
        old_balance_dest, new_balance_dest
    )

    prob      = model.predict_proba(input_df)[0][1]
    is_fraud  = prob >= THRESHOLD
    conf_pct  = prob * 100 if is_fraud else (1 - prob) * 100

    # ── Engineered feature summary ────────────────────────────────────────────
    ratio   = amount / (old_balance_orig + 1e-9) if old_balance_orig > 0 else 0
    zeroed  = int(new_balance_orig == 0 and old_balance_orig > 0)
    unmoved = int(old_balance_dest == new_balance_dest)

    if is_fraud:
        st.markdown(f"""
        <div class="result-card result-fraud">
            <div class="result-status">Classification Result</div>
            <div class="result-verdict">Fraudulent Transaction</div>
            <div class="result-confidence">Fraud probability — {prob*100:.1f}%  &nbsp;|&nbsp;  Threshold — {THRESHOLD:.2f}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill conf-bar-fraud" style="--bar-width:{prob*100:.1f}%; width:{prob*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <div class="result-status">Classification Result</div>
            <div class="result-verdict">Legitimate Transaction</div>
            <div class="result-confidence">Fraud probability — {prob*100:.1f}%  &nbsp;|&nbsp;  Threshold — {THRESHOLD:.2f}</div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill conf-bar-safe" style="--bar-width:{(1-prob)*100:.1f}%; width:{(1-prob)*100:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature signals ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:1.6rem">
        <div class="section-label">Feature Signals</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-tile">
            <span class="metric-val">{ratio:.3f}</span>
            <span class="metric-lbl">Amount Ratio</span>
        </div>
        <div class="metric-tile">
            <span class="metric-val">{"Yes" if zeroed else "No"}</span>
            <span class="metric-lbl">Balance Zeroed</span>
        </div>
        <div class="metric-tile">
            <span class="metric-val">{"Yes" if unmoved else "No"}</span>
            <span class="metric-lbl">Dest Unchanged</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Warning if suspicious signals ─────────────────────────────────────────
    warnings = []
    if zeroed:
        warnings.append("Origin balance dropped to zero — strong fraud indicator.")
    if unmoved:
        warnings.append("Destination balance unchanged — possible money mule pattern.")
    if ratio > 0.9:
        warnings.append(f"Amount ratio {ratio:.2f} — nearly full balance transferred.")

    if warnings and not is_fraud:
        warn_text = " &nbsp;/&nbsp; ".join(warnings)
        st.markdown(f"""
        <div class="warn-banner">
            Warning — suspicious signals detected despite low probability:<br>
            {warn_text}
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    PaySim Fraud Detection &nbsp;·&nbsp; AdaBoost Classifier &nbsp;·&nbsp; PR-AUC 0.97<br>
    Jordan University of Science and Technology &nbsp;·&nbsp; Big Data Analytics
</div>
""", unsafe_allow_html=True)
