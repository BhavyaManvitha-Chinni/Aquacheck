import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

st.set_page_config(page_title="AquaCheck", page_icon="💧", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #2196F3;'>💧 AquaCheck</h1>
    <h4 style='text-align: center; color: gray;'>AI-Powered Water Potability Analyzer</h4>
    <hr style='border:1px solid #ddd;'>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .result-card {
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        font-size: 18px;
        font-weight: 500;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .safe {
        border-left: 8px solid #4CAF50;
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .unsafe {
        border-left: 8px solid #F44336;
        background-color: #ffebee;
        color: #c62828;
    }
    .score {
        font-size: 22px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

FEATURES = [
    "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"
]

WHO_HELP = {
    "pH": "WHO guideline ~6.5–8.5",
    "Hardness": "Acceptable up to ~300 mg/L",
    "Solids": "Desirable < 500 ppm, up to 1500 ppm",
    "Chloramines": "Typical 2–4 ppm",
    "Sulfate": "≤ 500 mg/L",
    "Conductivity": "50–500 μS/cm typical",
    "Organic Carbon": "< 5–15 ppm",
    "Trihalomethanes": "< 100 μg/L",
    "Turbidity": "< 5 NTU desirable"
}

SAFE_COLOR = "#4CAF50"
UNSAFE_COLOR = "#F44336"

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("data/water_potability_clean.csv")
        rename_map = {
            "ph": "pH",
            "Organic_carbon": "Organic Carbon",
            "Trihalomethanes": "Trihalomethanes"
        }
        df = df.rename(columns=rename_map)
        needed = FEATURES + ["Potability"]
        return df[needed].copy()
    except:
        return None

df_clean = load_dataset()

def prob_to_risk(prob_safe: float) -> int:
    return int(round(prob_safe * 100))

def predict_proba(input_df: pd.DataFrame) -> float:
    return float(model.predict_proba(input_df[FEATURES].to_numpy())[0][1])

def probability_donut(prob_safe: float):
    probs = [prob_safe * 100, (1 - prob_safe) * 100]
    labels = ["Safe", "Unsafe"]
    colors = [SAFE_COLOR, UNSAFE_COLOR]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        probs, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    fig.gca().add_artist(centre_circle)
    ax.axis("equal")
    ax.set_title("Probability Breakdown")
    st.pyplot(fig)

def radar_chart_user_vs_means(input_df: pd.DataFrame):
    if df_clean is None: return
    safe_means = df_clean[df_clean["Potability"] == 1][FEATURES].mean()
    unsafe_means = df_clean[df_clean["Potability"] == 0][FEATURES].mean()
    user_vals = input_df[FEATURES].iloc[0]

    categories = FEATURES
    N = len(categories)

    def close_loop(values): return np.concatenate([values, values[:1]])

    combined = pd.DataFrame({
        "user": user_vals,
        "safe": safe_means,
        "unsafe": unsafe_means
    })
    vmax = combined.abs().max(axis=1).replace(0, 1e-9)
    norm = combined.divide(vmax, axis=0).T

    values_user = norm.loc["user"].values
    values_safe = norm.loc["safe"].values
    values_unsafe = norm.loc["unsafe"].values

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))
    ax.plot(angles, close_loop(values_user), linewidth=2, label="User", color="#2196F3")
    ax.fill(angles, close_loop(values_user), alpha=0.1, color="#2196F3")
    ax.plot(angles, close_loop(values_safe), linewidth=2, label="Safe Mean", color=SAFE_COLOR)
    ax.fill(angles, close_loop(values_safe), alpha=0.1, color=SAFE_COLOR)
    ax.plot(angles, close_loop(values_unsafe), linewidth=2, label="Unsafe Mean", color=UNSAFE_COLOR)
    ax.fill(angles, close_loop(values_unsafe), alpha=0.1, color=UNSAFE_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title("Your Input vs Safe/Unsafe Means")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)

if "history" not in st.session_state:
    st.session_state.history = []

def add_history(row: dict, max_len=20):
    st.session_state.history.insert(0, row)
    st.session_state.history = st.session_state.history[:max_len]

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Safe/Unsafe threshold", 0.0, 1.0, 0.30, 0.01)
st.sidebar.caption("Decision boundary for classifying Safe (1).")

st.sidebar.header("ℹ️ About AquaCheck")
st.sidebar.markdown(
    """
- **Model:** Random Forest (SMOTE + Calibration)  
- **Explainability:** Feature Importances  
- **UI:** Tabs, donut, radar chart, history  
- **Note:** WHO tooltips included
"""
)

tab_pred, tab_explain, tab_insights = st.tabs(
    ["🔮 Prediction", "🧠 Explainability", "📈 Dataset Insights"]
)

with tab_pred:
    st.subheader("🧪 Enter Water Quality Parameters")
    c1, c2, c3 = st.columns(3)

    with c1:
        ph = st.number_input("🌡️ pH", 0.0, 14.0, 7.0, 0.1, help=WHO_HELP["pH"])
        hardness = st.number_input("🪨 Hardness (mg/L)", 0.0, 1000.0, 150.0, help=WHO_HELP["Hardness"])
        solids = st.number_input("🧂 Solids (ppm)", 0.0, 100000.0, 20000.0, help=WHO_HELP["Solids"])
    with c2:
        chloramines = st.number_input("🧪 Chloramines (ppm)", 0.0, 20.0, 7.0, help=WHO_HELP["Chloramines"])
        sulfate = st.number_input("🧂 Sulfate (mg/L)", 0.0, 1000.0, 250.0, help=WHO_HELP["Sulfate"])
        conductivity = st.number_input("⚡ Conductivity (μS/cm)", 0.0, 2000.0, 400.0, help=WHO_HELP["Conductivity"])
    with c3:
        organic_carbon = st.number_input("🌿 Organic Carbon (ppm)", 0.0, 50.0, 15.0, help=WHO_HELP["Organic Carbon"])
        trihalomethanes = st.number_input("☣️ Trihalomethanes (μg/L)", 0.0, 300.0, 80.0, help=WHO_HELP["Trihalomethanes"])
        turbidity = st.number_input("🌫️ Turbidity (NTU)", 0.0, 50.0, 3.0, help=WHO_HELP["Turbidity"])

    st.markdown("---")
    if st.button("🔍 Check Potability", use_container_width=True):
        input_vals = [ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity]
        input_df = pd.DataFrame([input_vals], columns=FEATURES)

        prob_safe = predict_proba(input_df)
        pred = int(prob_safe >= threshold)
        risk = prob_to_risk(prob_safe)

        if pred == 1:
            st.markdown(f"<div class='result-card safe'>✅ Safe to Drink<br><div class='score'>Risk Score: {risk}/100</div>Confidence (Safe): {prob_safe*100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card unsafe'>⚠️ Unsafe to Drink<br><div class='score'>Risk Score: {risk}/100</div>Confidence (Unsafe): {(1 - prob_safe)*100:.2f}%</div>", unsafe_allow_html=True)

        probability_donut(prob_safe)
        radar_chart_user_vs_means(input_df)

        add_history({
            "pH": ph, "Hardness": hardness, "Solids": solids, "Chloramines": chloramines,
            "Sulfate": sulfate, "Conductivity": conductivity, "Organic Carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes, "Turbidity": turbidity,
            "Prob_Safe": round(prob_safe, 4), "Pred": pred, "Threshold": threshold, "RiskScore": risk
        })

    st.markdown("---")
    st.subheader("🕒 Recent Predictions (last 20)")
    if len(st.session_state.history) == 0:
        st.caption("No predictions yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

with tab_explain:
    st.subheader("🧠 Global Explainability")
    try:
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        importances.plot(kind="barh", ax=ax, color="#607D8B")
        ax.set_title("Random Forest Feature Importances")
        st.pyplot(fig)
    except:
        st.info("⚠️ Could not compute feature importances.")

with tab_insights:
    st.subheader("📈 Dataset Insights")
    if df_clean is None:
        st.info("Please add dataset at `data/water_potability_clean.csv`")
    else:
        counts = df_clean["Potability"].value_counts().sort_index()
        labels = ["Unsafe (0)", "Safe (1)"]
        vals = [counts.get(0, 0), counts.get(1, 0)]
        fig, ax = plt.subplots()
        ax.bar(labels, vals, color=[UNSAFE_COLOR, SAFE_COLOR])
        ax.set_title("Class Distribution")
        st.pyplot(fig)

        st.markdown("#### Summary Stats")
        st.dataframe(df_clean.describe().T)

        try:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_clean[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        except:
            st.caption("Heatmap not available.")
