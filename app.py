import base64
import json
import os
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="NutriScreen | Child Nutrition Screening",
                   page_icon="🌿", layout="wide")

# ---------- Optional hero photo (add hero.jpg to the repo) ----------
def hero_visual():
    if os.path.exists("hero.jpg"):
        with open("hero.jpg", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;height:100%;object-fit:cover;">'
    return '<span style="font-size:7rem;">👶</span>'

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@700&family=Poppins:wght@400;500;600;700;800&display=swap');
html {scroll-behavior: smooth;}
html, body, [class*="css"] {font-family: 'Poppins', sans-serif;}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 !important; max-width: 100% !important;}
.script {font-family: 'Caveat', cursive; color: #65a30d;}
.topbar {background:#1f2937; color:#fff; text-align:center; font-size:.8rem; padding:.5rem;}
.nav {display:flex; align-items:center; justify-content:space-between; padding:1rem 2.5rem;
      background:#fff; border-bottom:1px solid #eef0ec; flex-wrap:wrap; gap:.6rem;}
.logo {font-size:1.5rem; font-weight:700; color:#1f2937;}
.logo b {color:#65a30d;}
.nav-links a {color:#374151; text-decoration:none; font-weight:500; font-size:.92rem; margin:0 .9rem;}
.nav-links a:hover {color:#65a30d;}
.pill {display:inline-block; background:#84cc16; color:#fff !important; font-weight:600;
       padding:.75rem 1.6rem; border-radius:999px; text-decoration:none; box-shadow:0 4px 0 #65a30d;}
.pill.small {padding:.45rem 1.2rem; font-size:.85rem; box-shadow:none;}
.hero-grid {display:grid; grid-template-columns:1.1fr 1fr; gap:2rem; align-items:center;
            padding:3.5rem 2.5rem; background:linear-gradient(180deg,#f4f5f1 0%,#ffffff 100%);}
.hero-grid h1 {font-size:2.6rem; line-height:1.15; color:#1f2937; margin:.2rem 0 1rem 0; font-weight:800;}
.hero-grid .script-big {font-family:'Caveat',cursive; color:#65a30d; font-size:3.8rem; display:block; line-height:1;}
.hero-grid p {color:#4b5563; font-size:1.05rem; max-width:520px; margin-bottom:1.6rem;}
.hero-art {position:relative; width:380px; height:380px; margin:auto;}
.hero-circle {position:absolute; inset:15px; border-radius:50%; border:10px solid #84cc16;
              background:#ecfccb; display:flex; align-items:center; justify-content:center; overflow:hidden;}
.orb {position:absolute; width:92px; height:92px; border-radius:50%; border:5px solid #84cc16;
      background:#fff; font-size:2.6rem; display:flex; align-items:center; justify-content:center;
      box-shadow:0 4px 12px rgba(0,0,0,.12);}
.tape {position:absolute; bottom:6px; left:-20px; right:-20px; height:26px; border-radius:4px;
       background:repeating-linear-gradient(90deg,#facc15 0 18px,#eab308 18px 20px); transform:rotate(-6deg);
       box-shadow:0 3px 8px rgba(0,0,0,.18);}
.strip {display:flex; justify-content:center; gap:3rem; flex-wrap:wrap; padding:1.6rem 1rem; background:#84cc16; color:#fff;}
.strip div {text-align:center;} .strip b {display:block; font-size:1.8rem;} .strip span {font-size:.85rem; opacity:.95;}
.section {padding:3.5rem 2.5rem 1rem 2.5rem; max-width:1150px; margin:auto;}
.section-title {text-align:center; margin-bottom:2rem;}
.section-title .script {font-size:2rem; display:block; line-height:1;}
.section-title h2 {font-size:2.1rem; color:#1f2937; margin:.2rem 0 .6rem 0; font-weight:700;}
.section-title p {color:#6b7280; max-width:640px; margin:auto;}
.feature {text-align:center; padding:1rem;}
.feature .heart {font-size:2rem;} .feature h4 {margin:.4rem 0; color:#1f2937;}
.feature p {color:#6b7280; font-size:.93rem; margin:0;}
.res {border-radius:16px; padding:1.1rem 1.2rem; color:#fff;}
.res small {opacity:.92; font-size:.75rem; letter-spacing:.06em;}
.res h2 {color:#fff; margin:.15rem 0 .1rem 0; font-size:1.9rem; font-weight:800;}
.res span {font-size:.82rem; opacity:.95;}
.normal {background:#16a34a;} .moderate {background:#d97706;} .severe {background:#dc2626;}
.gauge {background:#e5e7eb; border-radius:999px; height:16px; overflow:hidden; margin:.4rem 0;}
.gauge div {height:100%; border-radius:999px;}
.alert {border-left:5px solid; border-radius:8px; padding:.7rem 1rem; margin:.5rem 0; font-size:.92rem;}
.alert.red {background:#fef2f2; border-color:#dc2626; color:#7f1d1d;}
.alert.amber {background:#fffbeb; border-color:#d97706; color:#78350f;}
.alert.blue {background:#eff6ff; border-color:#2563eb; color:#1e3a8a;}
.placeholder {background:#f7fee7; border:2px dashed #bef264; border-radius:16px;
              padding:2.5rem 1.5rem; text-align:center; color:#4d7c0f;}
.card {background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:1.3rem; height:100%;
       box-shadow:0 1px 3px rgba(0,0,0,.05);}
.card h4 {margin:0 0 .3rem 0; color:#1f2937;} .card p {margin:0; color:#6b7280; font-size:.93rem;}
.footer {background:#1f2937; color:#9ca3af; text-align:center; font-size:.82rem; padding:1.6rem; margin-top:3rem;}
div[data-testid="stForm"] {border:1px solid #e5e7eb; border-radius:16px; padding:1.2rem; background:#fff;}
button[kind="primaryFormSubmit"], button[kind="primary"] {border-radius:999px !important; font-weight:600;}
.pad {padding:0 2.5rem; max-width:1150px; margin:auto;}
@media (max-width: 850px) {
  .hero-grid {grid-template-columns:1fr; padding:2rem 1.2rem;}
  .hero-grid h1 {font-size:2rem;} .nav {padding:1rem;} .nav-links {display:none;}
  .section, .pad {padding-left:1rem; padding-right:1rem;}
  .hero-art {width:300px; height:300px;}
}
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

def section_title(script, title, sub=""):
    st.markdown(f'<div class="section-title"><span class="script">{script}</span>'
                f'<h2>{title}</h2><p>{sub}</p></div>', unsafe_allow_html=True)

# ---------- Top bar + navigation ----------
st.markdown(
    '<div class="topbar">Academic prototype · MIECRC framework · Not a medical device</div>'
    '<div class="nav"><div class="logo">🌿 Nutri<b>Screen</b></div>'
    '<div class="nav-links"><a href="#home">Home</a><a href="#screener">Screener</a>'
    '<a href="#results">Results</a><a href="#how">How it works</a><a href="#limits">Limitations</a></div>'
    '<a class="pill small" href="#screener">Try it now</a></div>',
    unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(
    '<div id="home"></div><div class="hero-grid"><div>'
    '<span class="script" style="font-size:1.8rem;">Smarter screening for</span>'
    '<h1>Spot <span class="script-big">hidden malnutrition</span> in children early</h1>'
    '<p>A multi-indicator machine learning tool that looks at age, weight, height and MUAC '
    'together, and compares its answer with the standard WHO MUAC-only rule.</p>'
    '<a class="pill" href="#screener">Try the Screener →</a></div>'
    '<div class="hero-art"><div class="hero-circle">' + hero_visual() + '</div>'
    '<div class="orb" style="top:0;right:-10px;">🥦</div>'
    '<div class="orb" style="top:130px;right:-40px;">🍎</div>'
    '<div class="orb" style="top:250px;right:-10px;">🥛</div>'
    '<div class="tape"></div></div></div>',
    unsafe_allow_html=True)

st.markdown(
    '<div class="strip"><div><b>0.89</b><span>ML macro-F1 (test)</span></div>'
    '<div><b>0.63</b><span>WHO rule macro-F1</span></div>'
    '<div><b>97%</b><span>At-risk children detected</span></div>'
    '<div><b>5,000</b><span>Training records</span></div></div>',
    unsafe_allow_html=True)

# ---------- Story ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
section_title("our story", "Why one measurement is not enough",
              "The WHO rule uses only mid-upper arm circumference. In our test set it missed "
              "105 at-risk children whose arm measurement looked normal.")
f1, f2, f3 = st.columns(3, gap="large")
f1.markdown('<div class="feature"><div class="heart">💚</div><h4>Multi-indicator</h4>'
            '<p>Combines age, weight, height and MUAC instead of relying on one number.</p></div>',
            unsafe_allow_html=True)
f2.markdown('<div class="feature"><div class="heart">💚</div><h4>Explainable</h4>'
            '<p>SHAP analysis shows which measurements drive each classification.</p></div>',
            unsafe_allow_html=True)
f3.markdown('<div class="feature"><div class="heart">💚</div><h4>Safe errors</h4>'
            '<p>No severe child was classified as normal in the held-out test set.</p></div>',
            unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Screener ----------
st.markdown('<div id="screener"></div><div class="section">', unsafe_allow_html=True)
section_title("try it yourself", "Child Nutritional Status Screener",
              "Enter the four measurements and compare both methods side by side.")
st.markdown('</div><div class="pad">', unsafe_allow_html=True)

left, right = st.columns([1, 1.25], gap="large")
with left:
    with st.form("child_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (months)", 0.0, 59.0, 24.0, 1.0)
        weight = c2.number_input("Weight (kg)", 1.0, 25.0, 9.0, 0.1)
        height = c1.number_input("Height / length (cm)", 40.0, 125.0, 80.0, 0.5)
        muac = c2.number_input("MUAC (cm)", 8.0, 20.0, 13.5, 0.1)
        submitted = st.form_submit_button("Classify child", use_container_width=True,
                                          type="primary")
    st.caption("Demo hidden case: age 6, weight 3.0, height 51.5, MUAC 11.99.")

with right:
    if not submitted:
        st.markdown('<div class="placeholder"><h3>Results appear here</h3>'
                    '<p>Enter the measurements and press <b>Classify child</b>.</p></div>',
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
                               "Uses age, weight, height, MUAC"), unsafe_allow_html=True)

        gcol = "#16a34a" if risk < 30 else "#d97706" if risk < 60 else "#dc2626"
        st.markdown(
            f'<div style="margin-top:1rem"><b>Risk score: {risk:.0f} / 100</b>'
            f' &nbsp;·&nbsp; ML confidence: <b>{confidence*100:.1f}%</b>'
            f'<div class="gauge"><div style="width:{risk:.0f}%; background:{gcol}"></div></div>'
            '<span style="color:#6b7280;font-size:.8rem">Risk score = model probability that the '
            'child is moderate or severe. Not clinically validated.</span></div>',
            unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"Probability (%)": proba * 100}, index=CLASSES), height=170)

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
            alert("blue", "Age is under 6 months. WHO MUAC cut-offs apply to 6 to 59 months.")
        out = [f for f, v in values.items()
               if not (TRAIN_RANGE[f][0] <= v <= TRAIN_RANGE[f][1])]
        if out:
            alert("amber", "Outside the training range, so the result may be unreliable: "
                  + ", ".join(out))
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Results ----------
st.markdown('<div id="results"></div><div class="section">', unsafe_allow_html=True)
section_title("the evidence", "Model performance",
              "Held-out test set of 1,000 children (710 normal, 220 moderate, 70 severe).")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Macro-F1", "0.892", "+0.259 vs WHO")
k2.metric("Severe recall", "82.9%", "+57.2 pts vs WHO")
k3.metric("Screening sensitivity", "96.9%", "+33.1 pts vs WHO")
k4.metric("Hidden cases found", "96 of 105", "WHO finds 0")
st.markdown('&nbsp;', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Macro-F1 by method**")
    st.bar_chart(pd.DataFrame({"Macro-F1": [0.633, 0.782, 0.869, 0.892]},
                 index=["WHO rule", "Decision Tree", "Random Forest", "XGBoost"]), height=280)
with c2:
    st.markdown("**Recall per class: WHO vs XGBoost (%)**")
    st.bar_chart(pd.DataFrame({"WHO rule": [99.9, 50.9, 25.7], "XGBoost": [97.7, 89.1, 82.9]},
                 index=["Normal", "Moderate", "Severe"]), height=280)
st.table(pd.DataFrame({
    "Method": ["XGBoost (selected)", "Random Forest", "Decision Tree", "WHO MUAC rule"],
    "Accuracy": ["94.8%", "93.9%", "88.8%", "83.9%"],
    "Macro-F1": [0.892, 0.869, 0.782, 0.633],
    "Moderate recall": ["89.1%", "86.4%", "71.4%", "50.9%"],
    "Severe recall": ["82.9%", "77.1%", "80.0%", "25.7%"]}).set_index("Method"))
st.markdown('</div>', unsafe_allow_html=True)

# ---------- How it works ----------
st.markdown('<div id="how"></div><div class="section">', unsafe_allow_html=True)
section_title("behind the scenes", "How it works")
h1, h2, h3 = st.columns(3, gap="medium")
h1.markdown('<div class="card"><h4>1 · WHO MUAC rule</h4><p>Looks at MUAC only: ≥ 12.5 cm normal, '
            '11.5 to &lt; 12.5 moderate, &lt; 11.5 severe. Simple, but blind to children whose '
            'arm circumference looks normal.</p></div>', unsafe_allow_html=True)
h2.markdown('<div class="card"><h4>2 · ML model</h4><p>XGBoost trained on age, weight, height and '
            'MUAC. It learns combinations, such as whether a weight is low for the child\'s age.</p></div>',
            unsafe_allow_html=True)
h3.markdown('<div class="card"><h4>3 · Confidence and risk</h4><p>The model gives a probability for '
            'each class. The highest is the confidence, and moderate plus severe forms the risk '
            'score. Low confidence triggers a referral warning.</p></div>', unsafe_allow_html=True)
st.markdown('&nbsp;', unsafe_allow_html=True)
st.markdown("**What drives the model (SHAP, overall importance)**")
st.bar_chart(pd.DataFrame({"Importance": [7.23, 5.86, 4.53, 2.07]},
             index=["MUAC", "Weight", "Age", "Height"]), height=240)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Limitations ----------
st.markdown('<div id="limits"></div><div class="section">', unsafe_allow_html=True)
section_title("please read", "Limitations")
st.markdown("""
- **Synthetic-looking dataset.** Some features are floored (weight at 3.0 kg for 32% of children) and class proportions are exactly 71 / 22 / 7.
- **Labels are not clinical ground truth.** The comparison measures agreement with dataset labels. The ML model sees four measurements and the WHO rule sees one.
- **No external validation.** One dataset, one split, no India-specific data.
- **Small severe class.** Only 70 severe children in the test set, so severe-class numbers are approximate.
- **Trade-off.** 16 healthy children flagged versus 1 for the WHO rule, in exchange for 96 hidden at-risk cases found.
- **Not a medical device.** Research prototype for decision support only.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">🌿 NutriScreen · MIECRC prototype · Topic 11: Machine Learning '
            'for Anthropometric Data Analysis · Academic use only</div>', unsafe_allow_html=True)
