"""
Streamlit application for the Lending Club Loan Default project.

This app provides multiple pages to explore the project:

* **Home** – a landing page with a high‑level overview.
* **Project Overview** – describes the business problem, data and methodology.
* **Make Prediction** – interactive form where users can enter borrower attributes and
  obtain a predicted probability of default using the trained model.
* **Model Performance** – displays evaluation metrics from the model comparison
  report.
* **Visualisations** – shows key charts from the exploratory data analysis.
* **Business Insights** – summarises key findings and recommendations.
* **About** – information about the project authorship.

The app loads a pre‑trained logistic regression model and a random forest model
along with a feature scaler.  It maps the input grade to an ordinal value,
scales the features and returns both the probability of default and the class
prediction.  Adjust the `MODEL_DIR` and `VISUALS_DIR` constants if you modify
the repository structure.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Constants
VISUALS_DIR = Path(__file__).resolve().parent.parent / "visuals"
DOCS_DIR = Path(__file__).resolve().parent.parent

# Predefined scaling parameters and logistic regression coefficients derived from a
# representative training run on a synthetic dataset that mimics the Lending Club
# data.  These values normalise input features and compute the default
# probability without loading external model files.
FEATURE_MINS = np.array([1000.39558, 20028.3940, 0.00192496, 5.00013842, 0.0, 1.0])
FEATURE_MAXS = np.array([34990.4009, 199986.469, 39.9960391, 29.9947343, 10.0, 7.0])
LR_INTERCEPT = -2.12363008
LR_COEFS = np.array([1.4633426, -0.09880605, 2.40813561, 2.29092991, 0.03520961, 1.90001884])


def map_grade_to_ordinal(grade: str) -> int:
    """Map credit grade letter to an ordinal integer (A=1 … G=7)."""
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    return mapping.get(grade.upper(), 1)


def predict_default(features: dict) -> float:
    """Predict default probability using a fixed logistic regression formula.

    The prediction uses min‑max scaling and fixed coefficients derived from the
    training run.  This avoids the need to load pickled models.

    Args:
        features: Dictionary containing borrower attributes.

    Returns:
        Estimated probability of default.
    """
    # Arrange features in the same order as used during training
    x = np.array([
        features["loan_amnt"],
        features["annual_inc"],
        features["dti"],
        features["int_rate"],
        features["emp_length"],
        map_grade_to_ordinal(features["grade"]),
    ], dtype=float)
    # Min‑max scaling
    x_scaled = (x - FEATURE_MINS) / (FEATURE_MAXS - FEATURE_MINS)
    # Logistic regression probability
    z = LR_INTERCEPT + np.dot(LR_COEFS, x_scaled)
    prob = 1.0 / (1.0 + np.exp(-z))
    return prob


def load_markdown_doc(filename: str) -> str:
    """Load a markdown document from the docs directory."""
    path = DOCS_DIR / filename
    try:
        return path.read_text()
    except Exception:
        return "Documentation not found."


def main() -> None:
    st.set_page_config(page_title="Lending Club Default Prediction", layout="centered")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Home",
            "Project Overview",
            "Make Prediction",
            "Model Performance",
            "Visualisations",
            "Business Insights",
            "About",
        ),
    )

    if page == "Home":
        st.title("Lending Club Loan Default Prediction")
        st.markdown(
            """
            Welcome to the Lending Club default prediction app.  This tool
            demonstrates a machine‑learning pipeline that uses historical loan
            data to estimate the probability that a borrower will default on a
            peer‑to‑peer loan.  Use the navigation bar on the left to learn
            about the project, explore the data, view model performance, and
            make your own predictions.
            """
        )

    elif page == "Project Overview":
        st.header("Project Overview")
        overview_md = load_markdown_doc("README.md")
        st.markdown(overview_md, unsafe_allow_html=True)

    elif page == "Make Prediction":
        st.header("Make a Prediction")
        st.markdown(
            "Fill in the borrower information below and click **Predict** to see
            the estimated probability of default using the logistic regression
            model.  Values should reflect the loan application at origination."
        )

        # Input widgets
        loan_amnt = st.number_input("Loan Amount (USD)", min_value=1000.0, max_value=40000.0, value=15000.0, step=500.0)
        annual_inc = st.number_input("Annual Income (USD)", min_value=10000.0, max_value=500000.0, value=75000.0, step=1000.0)
        dti = st.number_input("Debt‑to‑Income Ratio (%)", min_value=0.0, max_value=45.0, value=18.0, step=0.5)
        int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=13.0, step=0.1)
        emp_length = st.slider("Employment Length (years)", min_value=0, max_value=10, value=5)
        grade = st.selectbox("Credit Grade", options=list("ABCDEFG"), index=2)

        if st.button("Predict"):
            features = {
                "loan_amnt": loan_amnt,
                "annual_inc": annual_inc,
                "dti": dti,
                "int_rate": int_rate,
                "emp_length": emp_length,
                "grade": grade,
            }
            prob = predict_default(features)
            st.success(f"Estimated Probability of Default: {prob:.2%}")
            if prob > 0.5:
                st.error("High risk of default. Consider adjusting loan terms or requiring additional collateral.")
            else:
                st.info("Low to moderate risk of default.")

    elif page == "Model Performance":
        st.header("Model Performance")
        st.markdown("""The table below summarises the evaluation metrics for each trained model on the test set.
        These numbers are illustrative and derived from a synthetic dataset with similar characteristics to the real data.
        """)
        metrics_df = pd.DataFrame([
            {"Model": "Logistic Regression", "Accuracy": 0.825, "Precision": 0.844, "Recall": 0.963, "F1": 0.900, "ROC‑AUC": 0.776},
            {"Model": "Random Forest", "Accuracy": 0.818, "Precision": 0.839, "Recall": 0.961, "F1": 0.896, "ROC‑AUC": 0.746},
            {"Model": "XGBoost", "Accuracy": 0.812, "Precision": 0.844, "Recall": 0.944, "F1": 0.891, "ROC‑AUC": 0.736},
            {"Model": "LightGBM", "Accuracy": 0.817, "Precision": 0.849, "Recall": 0.942, "F1": 0.893, "ROC‑AUC": 0.733},
        ])
        st.dataframe(metrics_df.style.format({col: "{:.3f}" for col in metrics_df.columns if col != "Model"}))
        st.markdown(
            "These metrics are further discussed in the **Model Evaluation** and **Model Comparison** reports in the repository."
        )

    elif page == "Visualisations":
        st.header("Exploratory Visualisations")
        st.markdown("Below are key charts from the exploratory data analysis.  They provide insight into the distribution of numeric features, default rates by credit grade and correlations among variables.")
        st.subheader("Distribution of Loan Amount")
        st.image(str(VISUALS_DIR / "distribution_loan_amnt.png"), caption="Loan amount distribution (right‑skewed)")
        st.subheader("Default Rate by Credit Grade")
        st.image(str(VISUALS_DIR / "default_rate_by_grade.png"), caption="Higher grades have lower default rates")
        st.subheader("Correlation Heatmap")
        st.image(str(VISUALS_DIR / "correlation_heatmap.png"), caption="Correlation among numeric variables and default")

    elif page == "Business Insights":
        st.header("Business Insights & Recommendations")
        insights_md = load_markdown_doc("BUSINESS_RECOMMENDATIONS.md")
        st.markdown(insights_md, unsafe_allow_html=True)

    elif page == "About":
        st.header("About this Project")
        st.markdown(
            """
            This project was developed by **Rachel Oyeyemi** as part of a data analytics and
            AI portfolio.  It demonstrates an end‑to‑end workflow covering data
            acquisition, preprocessing, feature engineering, model training, evaluation
            and deployment via a Streamlit app.  The repository includes notebooks,
            scripts, documentation and a presentation, making it both resume‑ready
            and interview‑ready.

            The models used in this app are trained on a simulated dataset that
            approximates the characteristics of the publicly available Lending Club
            loan data.  For details on the data, modelling choices and business
            context, refer to the markdown reports in the repository.
            """
        )
