# 🥗 NutriScreen: Child Nutritional Status Screener

**Machine Learning for Anthropometric Data Analysis and Nutritional Status Classification**

NutriScreen is a web application that classifies a child's nutritional status (normal, moderate or severe) from four measurements: age, weight, height and MUAC. It shows the result of the standard WHO MUAC rule and a multi-indicator XGBoost model side by side, with a confidence value and a risk score.

> **Academic prototype.** Trained on a synthetic-looking dataset. It is not a medical device and must not be used for clinical decisions.

---

## Project background

The WHO MUAC rule uses a single measurement: MUAC of 12.5 cm or more is normal, 11.5 to below 12.5 cm is moderate, and below 11.5 cm is severe. Because it ignores age, weight and height, it cannot see a child whose arm circumference looks normal but whose other measurements indicate risk.

This project follows the **MIECRC framework** (Multi-Indicator Explainable ML Framework for Concurrent Nutritional Risk Classification), proposed after a review of 15 research papers (2020 to 2026) with Jain et al. (2022) as the base paper. It tests whether a model that combines several indicators detects more at-risk children than the single-indicator rule.

## Objectives

- Analyse an anthropometric dataset of 5,000 children
- Implement the WHO MUAC rule as a baseline
- Train and compare Decision Tree, Random Forest and XGBoost models
- Evaluate all methods on the same held-out test children
- Explain the best model with SHAP
- Deploy a web application showing the WHO result, ML result, confidence and risk score

## Dataset

Children Malnutrition Dataset (Kaggle, albertkingstone).

| Item | Details |
|---|---|
| Records | 5,000 children, no missing values, no duplicates |
| Classes | 71% normal (3,550), 22% moderate (1,100), 7% severe (350) |
| Features used | `age_months`, `weight_kg`, `height_cm`, `muac_cm` |
| Target | `nutrition_status` (normal / moderate / severe) |
| Dropped column | `bmi`, because 98.4% of its values were exactly 10.0 |

## Method

| Step | Details |
|---|---|
| Split | Stratified 80/20 (4,000 train, 1,000 test) |
| Models | Decision Tree, Random Forest (primary), XGBoost |
| Class imbalance | Balanced class weights (balanced sample weights for XGBoost) |
| Model selection | 5-fold stratified cross-validation on the training set only (XGBoost macro-F1 0.888, Random Forest 0.875, Decision Tree 0.808) |
| Evaluation | Accuracy, macro-F1, per-class recall, sensitivity and specificity, bootstrap 95% intervals, exact McNemar test |
| Explainability | SHAP (TreeExplainer) |

## Results (held-out test set, 1,000 children)

| Method | Accuracy | Macro-F1 | Moderate recall | Severe recall |
|---|---|---|---|---|
| **XGBoost (selected)** | 94.8% | 0.892 | 89.1% | 82.9% |
| Random Forest | 93.9% | 0.869 | 86.4% | 77.1% |
| Decision Tree | 88.8% | 0.782 | 71.4% | 80.0% |
| WHO MUAC rule | 83.9% | 0.633 | 50.9% | 25.7% |

- **Screening sensitivity** (moderate or severe vs normal): 96.9% for XGBoost vs 63.8% for the WHO rule. Specificity is 97.7% vs 99.9%.
- **Head-to-head:** of the children where the two methods disagreed, XGBoost was right for 137 and the WHO rule for 28 (exact McNemar p of about 2e-18).
- **Hidden cases:** 105 at-risk test children had MUAC of 12.5 cm or more. The WHO rule detects none of them by design, and XGBoost detected 96 (91.4%).
- **Safe errors:** 52 errors out of 1,000, all one step from the true class. No severe child was classified as normal.
- **Trade-off:** XGBoost flagged 16 healthy children, compared with 1 for the WHO rule.
- **Confidence:** mean 0.975 for correct predictions and 0.808 for incorrect ones.
- **Age check:** excluding children under 6 months did not change the result (macro-F1 0.627 for WHO, 0.905 for XGBoost).

## Explainability (SHAP)

MUAC is the most influential feature (7.23), followed by weight (5.86), age (4.53) and height (2.07). Age matters as context for weight and height, similar to weight-for-age logic. Height ranks last partly because it is highly correlated with age and weight (0.90 to 0.98).

## Web application features

- WHO MUAC classification and ML classification side by side, colour-coded
- ML confidence and a 0 to 100 risk score (the model's probability that the child is moderate or severe)
- Probability of each class
- Alerts for: disagreement between methods, a possible hidden case, confidence below 80%, age under 6 months, and inputs outside the training range
- Sections for model performance, how it works (with SHAP importance) and limitations

## Repository files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `nutrition_model.json` | Trained XGBoost model (XGBoost JSON format) |
| `model_info.json` | Feature names and class order |
| `requirements.txt` | Dependencies for deployment |
| `.streamlit/config.toml` | Light theme settings |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- The dataset appears synthetic: weight is floored at 3.0 kg for 32.4% of children, and class proportions are exactly 71 / 22 / 7.
- Labels are not clinical ground truth. The results show agreement with the dataset labels, and the ML model sees four measurements while the WHO rule sees one.
- The test set has only 70 severe children, so severe-class results are approximate.
- No external validation and no India-specific data.
- WHO MUAC cut-offs apply to children aged 6 to 59 months, but the dataset includes younger children.

## Future work

- Validate on real, clinically labelled data, ideally from Indian surveys, using WHO z-score classifications as the reference
- Choose a referral threshold by weighing missed cases against false alarms
- Pilot the tool with frontline health workers
- Extend to other indicators such as stunting and underweight

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Kaggle, GitHub

---

Academic project, Topic 11: Machine Learning for Anthropometric Data Analysis
