import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, section.main {
    background-color: #090b0f !important;
    color: #e8e6e1 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,255,178,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%,  rgba(0,180,255,0.04) 0%, transparent 50%),
        #090b0f !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 700px !important;
    padding: 3rem 1.8rem 6rem !important;
}

/* ── Force ALL inputs dark ── */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="select"] div,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #13161d !important;
    color: #e8e6e1 !important;
    border-color: rgba(255,255,255,0.1) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #63ffb2 !important;
}

/* Number input wrapper */
[data-testid="stNumberInput"] > div {
    background-color: #13161d !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}
[data-testid="stNumberInput"] > div:focus-within {
    border-color: rgba(99,255,178,0.45) !important;
    box-shadow: 0 0 0 3px rgba(99,255,178,0.07) !important;
}

/* Selectbox wrapper */
[data-testid="stSelectbox"] > div > div {
    background-color: #13161d !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e6e1 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(99,255,178,0.45) !important;
}

/* Dropdown popup */
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #13161d !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[role="option"], li {
    color: #e8e6e1 !important;
    font-family: 'DM Mono', monospace !important;
    background-color: #13161d !important;
}
[role="option"]:hover, li:hover {
    background-color: rgba(99,255,178,0.08) !important;
}

/* Stepper buttons on number input */
[data-testid="stNumberInput"] button {
    background-color: #13161d !important;
    color: rgba(232,230,225,0.5) !important;
    border-color: rgba(255,255,255,0.06) !important;
}
[data-testid="stNumberInput"] button:hover {
    background-color: rgba(99,255,178,0.1) !important;
    color: #63ffb2 !important;
}

/* Labels */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    color: rgba(232,230,225,0.45) !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #63ffb2 0%, #00c8ff 100%) !important;
    color: #090b0f !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-top: 1rem !important;
    box-shadow: 0 4px 24px rgba(99,255,178,0.18) !important;
    transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(99,255,178,0.28) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #63ffb2 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,255,178,0.2); border-radius: 99px; }

/* ── Animations ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.95) translateY(8px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes barGrow {
    from { width: 0; }
}

/* ── Header ── */
.hdr { text-align:center; padding:3rem 0 2.5rem; animation:fadeDown 0.7s ease both; }
.hdr-label {
    font-family:'DM Mono',monospace; font-size:0.68rem;
    letter-spacing:0.22em; color:#63ffb2; text-transform:uppercase;
    margin-bottom:1rem; opacity:0.8;
}
.hdr-title {
    font-size:clamp(2rem,5vw,3.2rem); font-weight:800;
    letter-spacing:-0.03em; line-height:1.1; color:#f0ede8;
}
.hdr-title span {
    background:linear-gradient(135deg,#63ffb2 0%,#00c8ff 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}

/* ── Section label ── */
.slbl {
    font-family:'DM Mono',monospace; font-size:0.65rem;
    letter-spacing:0.2em; color:rgba(99,255,178,0.55);
    text-transform:uppercase; margin-bottom:1rem;
    animation:fadeUp 0.6s ease both; animation-delay:0.2s;
}

/* ── Input wrapper card ── */
.icard {
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:1.6rem 1.6rem 1rem;
    margin-bottom:1rem;
    animation:fadeUp 0.7s ease both; animation-delay:0.3s;
    transition:border-color 0.3s;
}
.icard:hover { border-color:rgba(99,255,178,0.12); }

/* ── Result card ── */
.rcard {
    border-radius:14px; padding:2.2rem 1.8rem;
    text-align:center;
    animation:resultReveal 0.55s cubic-bezier(0.34,1.56,0.64,1) both;
    margin-top:1.4rem; position:relative; overflow:hidden;
}
.rcard-safe  { background:rgba(99,255,178,0.05); border:1px solid rgba(99,255,178,0.22); color:#63ffb2; }
.rcard-fraud { background:rgba(255,80,80,0.05);  border:1px solid rgba(255,80,80,0.22);  color:#ff5858; }
.rcard-lbl  { font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase; opacity:0.6; margin-bottom:0.7rem; }
.rcard-main { font-size:clamp(1.5rem,4vw,2.2rem); font-weight:800; letter-spacing:-0.02em; margin-bottom:0.5rem; }
.rcard-prob { font-family:'DM Mono',monospace; font-size:0.78rem; opacity:0.55; }
.bar-wrap   { margin-top:1.2rem; background:rgba(255,255,255,0.06); border-radius:99px; height:3px; overflow:hidden; }
.bar-safe   { height:100%; border-radius:99px; background:linear-gradient(90deg,#63ffb2,#00c8ff); animation:barGrow 0.8s cubic-bezier(0.22,1,0.36,1) both; animation-delay:0.3s; }
.bar-fraud  { height:100%; border-radius:99px; background:linear-gradient(90deg,#ff5858,#ff9a3c); animation:barGrow 0.8s cubic-bezier(0.22,1,0.36,1) both; animation-delay:0.3s; }

/* ── Signal tiles ── */
.sig-row { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:0.8rem; animation:fadeUp 0.5s ease both; animation-delay:0.5s; }
.sig-tile { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.9rem 0.7rem; text-align:center; }
.sig-val  { font-family:'DM Mono',monospace; font-size:1rem; font-weight:500; color:#63ffb2; display:block; margin-bottom:0.25rem; }
.sig-lbl  { font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:0.12em; text-transform:uppercase; color:rgba(232,230,225,0.3); }

/* ── Warning ── */
.warn { background:rgba(255,154,60,0.07); border:1px solid rgba(255,154,60,0.18); border-radius:10px; padding:0.85rem 1.1rem; font-family:'DM Mono',monospace; font-size:0.74rem; color:rgba(255,154,60,0.8); line-height:1.7; margin-top:0.8rem; animation:fadeUp 0.5s ease both; animation-delay:0.6s; }

/* ── Footer ── */
.ftr { text-align:center; font-family:'DM Mono',monospace; font-size:0.65rem; color:rgba(232,230,225,0.18); letter-spacing:0.08em; margin-top:4rem; line-height:1.9; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"] { display:none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
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
<div class="hdr">
    <div class="hdr-label">PaySim — Machine Learning</div>
    <div class="hdr-title">Fraud <span>Detection</span> System</div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.markdown(f"""
    <div class="warn">
        Model files not found. Ensure fraud_model.pkl, model_features.pkl,
        and best_threshold.pkl are in the same directory as app.py.<br>
        Error: {load_error}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="slbl">Transaction Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tx_type          = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    amount           = st.number_input("Amount",           min_value=0.0, value=50000.0,  step=1000.0)
    step             = st.number_input("Step (Hour)",      min_value=1,   max_value=744,  value=1,     step=1)
with col2:
    old_balance_orig = st.number_input("Old Balance — Origin",      min_value=0.0, value=80000.0, step=1000.0)
    new_balance_orig = st.number_input("New Balance — Origin",      min_value=0.0, value=30000.0, step=1000.0)
    old_balance_dest = st.number_input("Old Balance — Destination", min_value=0.0, value=0.0,     step=1000.0)
    new_balance_dest = st.number_input("New Balance — Destination", min_value=0.0, value=0.0,     step=1000.0)

predict_btn = st.button("Analyze Transaction")


# ── Feature builder ───────────────────────────────────────────────────────────
def build_features(tx_type, amount, step, old_orig, new_orig, old_dest, new_dest):
    row = {
        "step":                   step,
        "amount":                 amount,
        "oldbalanceOrg":          old_orig,
        "newbalanceOrig":         new_orig,
        "oldbalanceDest":         old_dest,
        "newbalanceDest":         new_dest,
        "type_CASH_OUT":          int(tx_type == "CASH_OUT"),
        "type_DEBIT":             0,
        "type_PAYMENT":           0,
        "type_TRANSFER":          int(tx_type == "TRANSFER"),
        "amount_ratio_orig":      amount / (old_orig + 1e-9) if old_orig > 0 else 0.0,
        "orig_balance_zeroed":    int(new_orig == 0 and old_orig > 0),
        "dest_balance_unchanged": int(old_dest == new_dest),
        "balance_error_orig":     abs(old_orig - amount - new_orig),
        "balance_error_dest":     abs(old_dest + amount - new_dest),
    }
    df = pd.DataFrame([row])
    if feature_cols:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[feature_cols]
    return df


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner(""):
        time.sleep(0.35)

    input_df = build_features(
        tx_type, amount, step,
        old_balance_orig, new_balance_orig,
        old_balance_dest, new_balance_dest
    )

    prob     = model.predict_proba(input_df)[0][1]
    is_fraud = prob >= THRESHOLD

    ratio   = amount / (old_balance_orig + 1e-9) if old_balance_orig > 0 else 0
    zeroed  = int(new_balance_orig == 0 and old_balance_orig > 0)
    unmoved = int(old_balance_dest == new_balance_dest)

    # ── Result card ──────────────────────────────────────────────────────────
    if is_fraud:
        st.markdown(f"""
        <div class="rcard rcard-fraud">
            <div class="rcard-lbl">Classification Result</div>
            <div class="rcard-main">Fraudulent Transaction</div>
            <div class="rcard-prob">Fraud probability — {prob*100:.1f}% &nbsp;|&nbsp; Threshold — {THRESHOLD:.2f}</div>
            <div class="bar-wrap"><div class="bar-fraud" style="width:{prob*100:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rcard rcard-safe">
            <div class="rcard-lbl">Classification Result</div>
            <div class="rcard-main">Legitimate Transaction</div>
            <div class="rcard-prob">Fraud probability — {prob*100:.1f}% &nbsp;|&nbsp; Threshold — {THRESHOLD:.2f}</div>
            <div class="bar-wrap"><div class="bar-safe" style="width:{(1-prob)*100:.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Signal tiles ─────────────────────────────────────────────────────────
    st.markdown('<div class="slbl" style="margin-top:1.6rem">Feature Signals</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sig-row">
        <div class="sig-tile">
            <span class="sig-val">{ratio:.3f}</span>
            <span class="sig-lbl">Amount Ratio</span>
        </div>
        <div class="sig-tile">
            <span class="sig-val">{"Yes" if zeroed  else "No"}</span>
            <span class="sig-lbl">Balance Zeroed</span>
        </div>
        <div class="sig-tile">
            <span class="sig-val">{"Yes" if unmoved else "No"}</span>
            <span class="sig-lbl">Dest Unchanged</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Warnings ─────────────────────────────────────────────────────────────
    warnings = []
    if zeroed:
        warnings.append("Origin balance dropped to zero — strong fraud indicator.")
    if unmoved:
        warnings.append("Destination balance unchanged — possible money mule pattern.")
    if ratio > 0.9:
        warnings.append(f"Amount ratio {ratio:.2f} — nearly full balance transferred.")

    if warnings and not is_fraud:
        st.markdown(f"""
        <div class="warn">
            Suspicious signals detected despite low fraud probability:<br>
            {"<br>".join(warnings)}
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ftr">
    PaySim Fraud Detection &nbsp;·&nbsp; AdaBoost &nbsp;·&nbsp; PR-AUC 0.97<br>

</div>
""", unsafe_allow_html=True)
