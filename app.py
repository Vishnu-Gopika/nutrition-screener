import json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="NutriScreen | Child Nutrition Screening",
                   page_icon="🩺", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg, #0f766e 0%, #115e59 55%, #134e4a 100%);
       color: white; padding: 2.2rem 2.4rem; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {color: white; font-size: 2.3rem; margin: 0 0 .4rem 0; font-weight: 800;}
.hero p {color: #ccfbf1; font-size: 1.05rem; margin: 0; max-width: 760px;}
.badge {display:inline-block; background: rgba(255,255,255,.15); color:#fff;
        padding: .25rem .8rem; border-radius: 999px; font-size: .8rem;
        margin-bottom: .8rem; letter-spacing: .04em;}
.stat-row {display:flex; gap:14px; flex-wrap:wrap; margin-top:1.3rem;}
.stat {background: rgba(255,255,255,.12); border-radius: 12px; padding: .7rem 1.1rem; min-width: 150px;}
.stat b {display:block; font-size: 1.5rem; color:#fff;}
.stat span {font-size: .8rem; color:#99f6e4;}
.card {background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:1.2rem 1.3rem;
       box-shadow: 0 1px 3px rgba(0,0,0,.06); height:100%;}
.card h4 {margin:0 0 .3rem 0; color:#0f172a;}
.card p {margin:0; color:#475569; font-size:.93rem;}
.res {border-radius:14px; padding:1.1rem 1.2rem; color:#fff;}
.res small {opacity:.9; font-size:.78rem; letter-spacing:.05em;}
.res h2 {color:#fff; margin:.15rem 0 .1rem 0; font-size:1.9rem; font-weight:800;}
.res span {font-size:.82rem; opacity:.92;}
.normal {background:#16a34a;} .moderate {background:#d97706;} .severe {background:#dc2626;}
.gauge {background:#e5e7eb; border-radius:999px; height:16px; overflow:hidden; margin:.4rem 0;}
.gauge div {height:100%; border-radius:999px;}
.alert {border-left:5px solid; border-radius:8px; padding:.7rem 1rem; margin:.5rem 0; font-size:.92rem;}
.alert.red {background:#fef2f2; border-color:#dc2626; color:#7f1d1d;}
.alert.amber {background:#fffbeb; border-color:#d97706; color:#78350f;}
.alert.blue {background:#eff6ff; border-color:#2563eb; color:#1e3a8a;}
.placeholder {background:#f8fafc; border:2px dashed #cbd5e1; border-radius:14px;
              padding:2.5rem 1.5rem; text-align:center; color:#64748b;}
.footer {text-align:center; color:#94a3b8; font-size:.8rem; margin-top:2rem;
         padding-top:1rem; border-top:1px solid #e5e7eb;}
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("nutrition_model.json")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_model()
FEATURES = info["features"]
CLASSES = info["classes"]
LOW_CONFIDENCE = 0.80
TRAIN_RANGE = {"age_months": (1, 60), "weight_kg": (3.0, 10.8),
               "height_cm": (45, 105), "muac_cm": (10.2, 15.8)}

def who_classify(muac):
    if muac >= 12.5:
        return "normal"
    if muac >= 11.5:
        return "moderate"
    return "severe"

def result_card(title, status, note):
    return (f'<div class="res {status}"><small>{title}</small>'
            f'<h2>{status.upper()}</h2><span>{note}</span></div>')

def alert(kind, text):
    st.markdown(f'<div class="alert {kind}">{text}</div>', unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(
    '<div class="hero">'
    '<div class="badge">MIECRC FRAMEWORK · ACADEMIC PROTOTYPE</div>'
    '<h1>🩺 NutriScreen</h1>'
    '<p>Multi-indicator machine learning for classifying child nutritional status, '
    'compared side by side with the WHO MUAC-only rule.</p>'
    '<div class="stat-row">'
    '<div class="stat"><b>0.89</b><span>ML macro-F1 (test)</span></div>'
    '<div class="stat"><b>0.63</b><span>WHO rule macro-F1</span></div>'
    '<div class="stat"><b>97%</b><span>At-risk children detected</span></div>'
    '<div class="stat"><b>5,000</b><span>Training records</span></div>'
    '</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🩺 Screener", "📊 Model Performance", "🔍 How It Works", "⚠️ Limitations"])

# ================= TAB 1: SCREENER =================
with tab1:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown("#### Enter the child's measurements")
        with st.form("child_form"):
            c1, c2 = st.columns(2)
            age = c1.number_input("Age (months)", 0.0, 59.0, 24.0, 1.0)
            weight = c2.number_input("Weight (kg)", 1.0, 25.0, 9.0, 0.1)
            height = c1.number_input("Height / length (cm)", 40.0, 125.0, 80.0, 0.5)
            muac = c2.number_input("MUAC (cm)", 8.0, 20.0, 13.5, 0.1)
            submitted = st.form_submit_button("Classify child", use_container_width=True,
                                              type="primary")
        st.caption("Demo values for a hidden case: age 6, weight 3.0, height 51.5, MUAC 11.99.")

    with right:
        if not submitted:
            st.markdown(
                '<div class="placeholder"><h3>Results appear here</h3>'
                '<p>Enter the four measurements and press <b>Classify child</b>.</p></div>',
                unsafe_allow_html=True)
        else:
            values = {"age_months": age, "weight_kg": weight,
                      "height_cm": height, "muac_cm": muac}
            X_new = pd.DataFrame([values])[FEATURES]
            proba = model.predict_proba(X_new)[0]
            pred = int(proba.argmax())
            ml_status = CLASSES[pred]
            confidence = float(proba[pred])
            risk = float(proba[1] + proba[2]) * 100
            who_status = who_classify(muac)

            a, b = st.columns(2)
            a.markdown(result_card("WHO MUAC RULE", who_status, "Uses MUAC only"),
                       unsafe_allow_html=True)
            b.markdown(result_card("ML MODEL (XGBOOST)", ml_status,
                                   "Uses age, weight, height, MUAC"),
                       unsafe_allow_html=True)

            gcol = "#16a34a" if risk < 30 else "#d97706" if risk < 60 else "#dc2626"
            st.markdown(
                f'<div style="margin-top:1rem"><b>Risk score: {risk:.0f} / 100</b> '
                f'&nbsp;·&nbsp; ML confidence: <b>{confidence*100:.1f}%</b>'
                f'<div class="gauge"><div style="width:{risk:.0f}%; background:{gcol}"></div></div>'
                '<span style="color:#64748b;font-size:.8rem">Risk score = model probability '
                'that the child is moderate or severe. Not clinically validated.</span></div>',
                unsafe_allow_html=True)

            prob_df = pd.DataFrame({"Probability (%)": proba * 100}, index=CLASSES)
            st.bar_chart(prob_df, height=180)

            if ml_status != "normal" and who_status == "normal":
                alert("red", "<b>Possible hidden case.</b> MUAC looks normal, but the other "
                      "measurements suggest risk. Refer for full assessment.")
            elif who_status != ml_status:
                alert("amber", f"<b>The methods disagree.</b> WHO rule: {who_status}. "
                      f"ML model: {ml_status}. The ML model also uses age, weight and height.")
            if confidence < LOW_CONFIDENCE:
                alert("amber", f"<b>Low confidence</b> (below {LOW_CONFIDENCE:.0%}). "
                      "Treat as uncertain and refer for clinical review.")
            if age < 6:
                alert("blue", "Age is under 6 months. WHO MUAC cut-offs apply to children "
                      "aged 6 to 59 months.")
            out = [f for f, v in values.items()
                   if not (TRAIN_RANGE[f][0] <= v <= TRAIN_RANGE[f][1])]
            if out:
                alert("amber", "Outside the training range, so the result may be unreliable: "
                      + ", ".join(out))

    st.markdown("&nbsp;")
    alert("blue", "<b>Academic prototype.</b> Trained on a synthetic dataset. "
          "Not a medical device. Do not use for clinical decisions.")

# ================= TAB 2: PERFORMANCE =================
with tab2:
    st.markdown("#### Held-out test set: 1,000 children (710 normal, 220 moderate, 70 severe)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Macro-F1", "0.892", "+0.259 vs WHO")
    k2.metric("Severe recall", "82.9%", "+57.2 pts vs WHO")
    k3.metric("Screening sensitivity", "96.9%", "+33.1 pts vs WHO")
    k4.metric("Hidden cases found", "96 of 105", "WHO finds 0")

    st.markdown("&nbsp;")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Macro-F1 by method**")
        st.bar_chart(pd.DataFrame(
            {"Macro-F1": [0.633, 0.782, 0.869, 0.892]},
            index=["WHO rule", "Decision Tree", "Random Forest", "XGBoost"]), height=300)
    with c2:
        st.markdown("**Recall per class: WHO vs XGBoost (%)**")
        st.bar_chart(pd.DataFrame(
            {"WHO rule": [99.9, 50.9, 25.7], "XGBoost": [97.7, 89.1, 82.9]},
            index=["Normal", "Moderate", "Severe"]), height=300)

    st.markdown("**Full comparison**")
    st.table(pd.DataFrame({
        "Method": ["XGBoost (selected)", "Random Forest", "Decision Tree", "WHO MUAC rule"],
        "Accuracy": ["94.8%", "93.9%", "88.8%", "83.9%"],
        "Macro-F1": [0.892, 0.869, 0.782, 0.633],
        "Moderate recall": ["89.1%", "86.4%", "71.4%", "50.9%"],
        "Severe recall": ["82.9%", "77.1%", "80.0%", "25.7%"],
    }).set_index("Method"))
    st.caption("XGBoost and Random Forest are statistically comparable (overlapping bootstrap "
               "intervals). XGBoost was chosen by cross-validation. No severe child was "
               "classified as normal.")

# ================= TAB 3: HOW IT WORKS =================
with tab3:
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown('<div class="card"><h4>1 · WHO MUAC rule</h4><p>Looks at MUAC only. '
                '≥ 12.5 cm normal, 11.5 to &lt; 12.5 cm moderate, &lt; 11.5 cm severe. '
                'Simple, but it cannot see children whose arm circumference looks normal.</p></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="card"><h4>2 · ML model</h4><p>XGBoost trained on age, weight, '
                'height and MUAC. It learns combinations of measurements, for example whether a '
                'weight is low for the child\'s age.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><h4>3 · Confidence and risk</h4><p>The model outputs a '
                'probability for each class. The highest is the confidence, and the chance of '
                'moderate or severe forms the risk score. Low confidence triggers a referral '
                'warning.</p></div>', unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.markdown("#### What drives the model (SHAP analysis)")
    st.bar_chart(pd.DataFrame(
        {"Overall importance": [7.23, 5.86, 4.53, 2.07]},
        index=["MUAC", "Weight", "Age", "Height"]), height=260)
    st.caption("MUAC matters most. Age matters through context: a given weight or height is more "
               "or less concerning depending on age. Height ranks last partly because it is "
               "highly correlated with age and weight.")

# ================= TAB 4: LIMITATIONS =================
with tab4:
    st.markdown("#### Please read before interpreting results")
    st.markdown("""
- **Synthetic-looking dataset.** Some features are floored (weight at 3.0 kg for 32% of children), and class proportions are exactly 71 / 22 / 7.
- **Labels are not clinical ground truth.** The comparison measures agreement with the dataset labels. The ML model sees four measurements, the WHO rule sees one, so part of its advantage comes from extra information.
- **No external validation.** One dataset, one split. No data specific to India.
- **Small severe class.** Only 70 severe children in the test set, so severe-class numbers are approximate.
- **Trade-off.** The model flags 16 healthy children versus 1 for the WHO rule, in exchange for catching 96 hidden at-risk cases.
- **Age range.** WHO MUAC cut-offs apply to 6 to 59 months. The dataset includes younger children, though excluding them did not change the result.
- **Not a medical device.** It is a research prototype for decision support only.
""")

st.markdown('<div class="footer">NutriScreen · MIECRC prototype · Topic 11: Machine Learning '
            'for Anthropometric Data Analysis · Academic use only</div>',
            unsafe_allow_html=True)
