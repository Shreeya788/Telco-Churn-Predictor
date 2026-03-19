import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #8892a4 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4b5a7a !important;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2130;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161f;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2130;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #6b7a96;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 20px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2640 !important;
    color: #7eb8ff !important;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    padding: 14px 0;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 24px rgba(37,99,235,0.35);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(37,99,235,0.5);
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}

/* Cards */
.result-card {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 16px;
}
.risk-high {
    border-left: 4px solid #ef4444;
    background: linear-gradient(135deg, #13161f 0%, #1f1318 100%);
}
.risk-low {
    border-left: 4px solid #22c55e;
    background: linear-gradient(135deg, #13161f 0%, #121b14 100%);
}

/* Metric boxes */
.metric-box {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b5a7a;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8eaf0;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.78rem;
    color: #6b7a96;
    margin-top: 4px;
}

/* Risk badge */
.badge-high {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-low {
    display: inline-block;
    background: rgba(34,197,94,0.12);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* Page title */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8eaf0;
    letter-spacing: -0.02em;
}
.page-subtitle {
    color: #4b5a7a;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* Divider */
hr {
    border-color: #1e2130 !important;
    margin: 1.5rem 0 !important;
}

/* Streamlit metric */
[data-testid="stMetric"] {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #6b7a96 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #e8eaf0 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Info / warning boxes */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD ASSETS ────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model    = pickle.load(open("model.pkl",    "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    scaler   = pickle.load(open("scaler.pkl",   "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    return model, encoders, scaler, features

try:
    model, encoders, scaler, features = load_assets()
    assets_loaded = True
except Exception:
    assets_loaded = False


# ─── SIDEBAR INPUTS ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="page-title" style="font-size:1.3rem;">⚙️ Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Fill in the details below</div>', unsafe_allow_html=True)

    st.markdown("### 👤 Demographics")
    gender     = st.selectbox("Gender",         ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Has Partner",    ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure     = st.slider("Tenure (months)", 0, 72, 12,
                           help="How long the customer has been with the company")

    st.markdown("### 📡 Services")
    phone_service    = st.selectbox("Phone Service",    ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines",   ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security",  ["Yes", "No", "No internet service"])
    online_backup    = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support     = st.selectbox("Tech Support",     ["Yes", "No", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("### 💳 Billing")
    contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",   0.0, 9000.0, 1000.0, step=10.0)

    predict_clicked = st.button("🔍 Predict Churn", use_container_width=True)


# ─── MAIN AREA ──────────────────────────────────────────────────
st.markdown('<div class="page-title">📉 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Predict churn risk and understand revenue exposure — before it\'s too late.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Prediction Results", "📊 Feature Importance"])


# ─── GAUGE CHART HELPER ─────────────────────────────────────────
def make_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.6), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#13161f")
    ax.set_facecolor("#13161f")

    # Arc segments
    angles = np.linspace(np.pi, 0, 200)
    lw = 22

    # Background track
    ax.plot(np.cos(angles), np.sin(angles), color="#1e2130", linewidth=lw, solid_capstyle="butt")

    # Colored fill up to prob
    fill_angles = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    color = "#ef4444" if prob >= 0.6 else "#f59e0b" if prob >= 0.35 else "#22c55e"
    ax.plot(np.cos(fill_angles), np.sin(fill_angles), color=color, linewidth=lw, solid_capstyle="butt")

    # Needle
    needle_angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.62 * np.cos(needle_angle), 0.62 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#e8eaf0",
                                lw=2, mutation_scale=14))
    ax.plot(0, 0, "o", color="#e8eaf0", markersize=7, zorder=5)

    # Labels
    ax.text(-1.05, -0.22, "0%",  color="#4b5a7a", fontsize=9, ha="center", fontfamily="monospace")
    ax.text( 0,    1.18,  "50%", color="#4b5a7a", fontsize=9, ha="center", fontfamily="monospace")
    ax.text( 1.05, -0.22, "100%",color="#4b5a7a", fontsize=9, ha="center", fontfamily="monospace")

    # Center percentage
    ax.text(0, -0.38, f"{prob*100:.1f}%", color=color,
            fontsize=26, ha="center", va="center",
            fontweight="bold", fontfamily="monospace")
    ax.text(0, -0.62, "Churn Probability", color="#6b7a96",
            fontsize=8.5, ha="center", va="center")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.75, 1.3)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ─── TAB 1: PREDICTION ──────────────────────────────────────────
with tab1:
    if not assets_loaded:
        st.info("⚠️ Model files not found. Run `train.py` first to generate `model.pkl`, `encoders.pkl`, `scaler.pkl`, and `features.pkl`.")

    elif not predict_clicked:
        # Placeholder state
        st.markdown("""
        <div class="result-card" style="text-align:center; padding: 60px 32px;">
            <div style="font-size:3rem; margin-bottom:12px;">🎯</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700; color:#e8eaf0;">
                Ready to predict
            </div>
            <div style="color:#4b5a7a; font-size:0.88rem; margin-top:8px;">
                Configure the customer profile in the sidebar, then hit <strong style="color:#7eb8ff;">Predict Churn</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Build input ──
        input_dict = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([input_dict])

        for col in input_df.select_dtypes(include="object").columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        prob    = model.predict_proba(input_df[features])[0][1]
        churned = prob >= 0.5

        annual_revenue  = monthly_charges * 12
        risk_level      = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.35 else "LOW"
        badge_class     = "badge-high" if churned else "badge-low"
        card_class      = "risk-high"  if churned else "risk-low"

        # ── Layout: gauge + metrics ──
        col_gauge, col_metrics = st.columns([1, 1], gap="large")

        with col_gauge:
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="margin-bottom:10px;">
                    <span class="{badge_class}">{risk_level} RISK</span>
                </div>
                <div style="font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:#e8eaf0; margin-bottom:4px;">
                    {"⚠️ This customer is likely to churn" if churned else "✅ This customer looks retained"}
                </div>
                <div style="color:#4b5a7a; font-size:0.82rem;">
                    Model confidence · threshold @ 50%
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig = make_gauge(prob)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

        with col_metrics:
            st.markdown("#### 💰 Revenue Impact")
            st.metric("Annual Revenue at Risk", f"${annual_revenue:,.2f}",
                      delta=f"−{prob*100:.1f}% retention probability",
                      delta_color="inverse")

            expected_loss = annual_revenue * prob
            st.metric("Expected Revenue Loss", f"${expected_loss:,.2f}",
                      help="Annual revenue × churn probability")

            st.markdown("<br>", unsafe_allow_html=True)

            if churned:
                st.error(f"💡 **Intervention recommended** — retaining this customer protects **${annual_revenue:,.2f}/year**.")

                # Risk factors quick summary
                flags = []
                if contract == "Month-to-month":     flags.append("Month-to-month contract")
                if internet_service == "Fiber optic": flags.append("Fiber optic internet")
                if tenure < 12:                       flags.append("Low tenure (< 12 mo)")
                if paperless == "Yes":                flags.append("Paperless billing")
                if payment_method == "Electronic check": flags.append("Electronic check payment")
                if online_security == "No":           flags.append("No online security")

                if flags:
                    st.markdown("**Common churn indicators detected:**")
                    for f in flags:
                        st.markdown(f"- 🔴 {f}")
            else:
                st.success("Customer shows strong retention signals. Continue monitoring.")

                positives = []
                if contract != "Month-to-month": positives.append(f"{contract} contract")
                if tenure >= 24:                 positives.append(f"Long tenure ({tenure} mo)")
                if online_security == "Yes":     positives.append("Online Security active")
                if tech_support == "Yes":        positives.append("Tech Support active")

                if positives:
                    st.markdown("**Positive retention signals:**")
                    for p in positives:
                        st.markdown(f"- 🟢 {p}")

        # ── Input summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View input summary", expanded=False):
            summary_cols = st.columns(4)
            fields = list(input_dict.items())
            per_col = (len(fields) + 3) // 4
            for i, col in enumerate(summary_cols):
                for k, v in fields[i*per_col:(i+1)*per_col]:
                    col.markdown(f"<div style='font-size:0.75rem;color:#4b5a7a;text-transform:uppercase;letter-spacing:.06em'>{k}</div>"
                                 f"<div style='font-size:0.88rem;color:#e8eaf0;margin-bottom:10px'>{v}</div>",
                                 unsafe_allow_html=True)


# ─── TAB 2: FEATURE IMPORTANCE ──────────────────────────────────
with tab2:
    st.markdown("#### Top Features Driving Churn")
    st.markdown('<div style="color:#4b5a7a;font-size:0.86rem;margin-bottom:20px;">Based on model-derived feature importances from training data.</div>', unsafe_allow_html=True)
    try:
        st.image("feature_importance.png", width='stretch')
    except Exception:
        st.info("Run `train.py` first to generate the feature importance chart.")