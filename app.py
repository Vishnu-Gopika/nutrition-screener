import json
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Child Nutritional Status Screener",
                   page_icon="🩺", layout="centered")

# ----- Load model and settings (cached so it loads once) -----
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("nutrition_model.json")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_model()
FEATURES = info["features"]
CLASSES = info["classes"]          # ['normal', 'moderate', 'severe']

LOW_CONFIDENCE = 0.80              # below this, advise clinical review

# Ranges seen in the training data (used for out-of-range warnings)
TRAIN_RANGE = {
    "age_months": (1, 60), "weight_kg": (3.0, 10.8),
    "height_cm": (45, 105), "muac_cm": (10.2, 15.8),
}

# ----- WHO MUAC rule -----
def who_classify(muac_cm):
    if muac_cm >= 12.5:
        return "normal"
    if muac_cm >= 11.5:
        return "moderate"
    return "severe"

def show_status(label, status):
    text = f"**{label}: {status.upper()}**"
    if status == "normal":
        st.success(text)
    elif status == "moderate":
        st.warning(text)
    else:
        st.error(text)

# ----- Page header -----
st.title("🩺 Child Nutritional Status Screener")
st.caption("MIECRC prototype: WHO MUAC rule vs multi-indicator machine learning")
st.info("Academic prototype trained on a synthetic dataset. "
        "It is **not** a medical device and must not be used for clinical decisions.")

# ----- Input form -----
with st.form("child_form"):
    st.subheader("Child's measurements")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (months)", min_value=0.0, max_value=59.0,
                          value=24.0, step=1.0)
    weight = c2.number_input("Weight (kg)", min_value=1.0, max_value=25.0,
                             value=9.0, step=0.1)
    height = c1.number_input("Height / length (cm)", min_value=40.0,
                             max_value=125.0, value=80.0, step=0.5)
    muac = c2.number_input("MUAC (cm)", min_value=8.0, max_value=20.0,
                           value=13.5, step=0.1)
    submitted = st.form_submit_button("Classify", use_container_width=True)

# ----- Results -----
if submitted:
    values = {"age_months": age, "weight_kg": weight,
              "height_cm": height, "muac_cm": muac}
    X_new = pd.DataFrame([values])[FEATURES]

    proba = model.predict_proba(X_new)[0]
    pred = int(proba.argmax())
    ml_status = CLASSES[pred]
    confidence = float(proba[pred])
    risk_score = float(proba[1] + proba[2]) * 100   # P(moderate or severe)
    who_status = who_classify(muac)

    st.divider()
    st.subheader("Results")
    col_a, col_b = st.columns(2)
    with col_a:
        show_status("WHO MUAC rule", who_status)
        st.caption("Uses MUAC only")
    with col_b:
        show_status("ML model (XGBoost)", ml_status)
        st.caption("Uses age, weight, height and MUAC")

    m1, m2 = st.columns(2)
    m1.metric("ML confidence", f"{confidence*100:.1f}%")
    m2.metric("Risk score (at risk)", f"{risk_score:.0f} / 100")
    st.progress(min(int(round(risk_score)), 100))
    st.caption("Risk score = the model's probability that the child is moderate "
               "or severe. It is not a clinically validated score.")

    # Class probabilities
    st.markdown("**Probability of each class**")
    prob_df = pd.DataFrame({"Class": CLASSES, "Probability": proba}).set_index("Class")
    st.bar_chart(prob_df)

    # ----- Messages and warnings -----
    if who_status != ml_status:
        st.warning(f"The two methods disagree: WHO rule says **{who_status}**, "
                   f"the ML model says **{ml_status}**. The ML model also uses "
                   "age, weight and height, which the WHO rule ignores.")
    if ml_status != "normal" and who_status == "normal":
        st.error("Possible hidden case: MUAC looks normal, but the other "
                 "measurements suggest risk. Refer for full assessment.")
    if confidence < LOW_CONFIDENCE:
        st.warning(f"Low confidence (below {LOW_CONFIDENCE:.0%}). "
                   "Treat this result as uncertain and refer for clinical review.")
    if age < 6:
        st.warning("Age is under 6 months. The WHO MUAC cut-offs apply to "
                   "children aged 6 to 59 months.")
    out_of_range = [f for f, v in values.items()
                    if not (TRAIN_RANGE[f][0] <= v <= TRAIN_RANGE[f][1])]
    if out_of_range:
        st.warning("These values are outside the range the model was trained on, "
                   "so the result may be unreliable: " + ", ".join(out_of_range))

with st.expander("About this tool"):
    st.markdown(
        "- **WHO rule:** MUAC ≥ 12.5 cm normal; 11.5 to <12.5 moderate; <11.5 severe.\n"
        "- **ML model:** XGBoost trained on 5,000 records (macro-F1 0.89 on a held-out test set, vs 0.63 for the WHO rule).\n"
        "- **Limits:** synthetic dataset, labels are not clinical ground truth, no external validation."
    )