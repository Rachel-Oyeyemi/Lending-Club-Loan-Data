# Project Charter – Lending Club Loan Data

## Business Problem

Lending Club wants to identify borrowers who are likely to default so that it can manage credit risk, set appropriate interest rates and adjust underwriting policies.  Historically, many loans were issued without leveraging predictive analytics, leading to higher default rates and reduced returns for investors.  By analysing historical loan data, we aim to build a model that predicts the probability of default for new applicants and provides insights into the key drivers of credit risk.

## Project Objectives

1. **Data Understanding & Preparation** – Acquire and clean Lending Club’s loan dataset, handle missing values and encode categorical variables.  Reduce dimensionality by dropping columns with high missingness and derive meaningful features.
2. **Model Development** – Train baseline and advanced machine‑learning models to classify loans as `default` or `fully paid`.  Evaluate models using appropriate metrics and select the best model for deployment.
3. **Insight Generation** – Conduct exploratory data analysis and feature importance analysis to identify the factors most associated with default.  Summarise the findings in a business‑friendly report.
4. **Deployment Prototype** – Build an interactive web application that allows users to explore the data, view model performance, and input borrower information to obtain default‑probability predictions.
5. **Documentation & Presentation** – Produce comprehensive documentation including a project charter, EDA report, model comparison, evaluation report and business recommendations.  Prepare an executive‑level slide deck to communicate the project results.

## Stakeholders

- **Risk Management Team** – Responsible for setting credit policy and monitoring portfolio performance.  Primary consumers of the predictive model.
- **Product & Underwriting Team** – Use model insights to adjust loan terms, interest rates and credit limits.
- **Investors** – Interested in the expected return and risk profile of the loan portfolio.
- **Regulators & Compliance** – Ensure that the model adheres to fair‑lending laws and data privacy regulations.
- **Data Science Team** – Executes the project, maintains models and monitors performance.

## Success Metrics

- **Model performance:**  Achieve an ROC‑AUC ≥ 0.90 and F1‑score ≥ 0.75 on the test set, balancing recall and precision.
- **Business outcomes:**  Reduction in default rate by ≥ 5 % relative to historical baseline when applied to new loans, without significantly reducing approval volume.
- **Adoption:**  Risk managers integrate the model into the underwriting workflow and use the Streamlit app for decision support.
- **Compliance:**  Model documentation meets regulatory requirements, and fairness metrics show no undue bias against protected groups.

## Expected Business Impact

Implementing the model is expected to:

- Reduce financial losses from defaulted loans by proactively identifying high‑risk borrowers.
- Enable risk‑based pricing and credit limit adjustments, improving profitability.
- Streamline underwriting by automating part of the risk assessment process and focusing human effort on ambiguous cases.
- Provide transparency and interpretability through feature importance and partial dependence analyses, supporting regulatory compliance.

## Technical Architecture

The solution consists of the following components:

- **Data Storage:**  Raw data stored in `data/raw/` and processed data in `data/processed/`.  Version control via GitHub ensures reproducibility.
- **Data Engineering Scripts:**  Python modules (`download_data.py`, `preprocess.py`, `feature_engineering.py`) handle downloading, cleaning and feature creation.
- **Modelling Pipeline:**  Scripts (`train_model.py`, `evaluate_model.py`) perform train/test splits, model training and evaluation.  Models are saved as pickled objects in `models/`.
- **Visualization & Reporting:**  Notebooks in `notebooks/` produce EDA and modeling analyses.  Reports are written in Markdown and stored at the repository root.
- **Streamlit Application:**  Provides a user interface to explore the project and generate predictions using the trained model.
- **CI/CD (future):**  GitHub Actions or other tooling can automate data ingestion, retraining and deployment.

## End‑to‑End Workflow

1. **Data Acquisition** – Download the Kaggle dataset using the Kaggle API and save it in `data/raw/`.
2. **Preprocessing** – Run `preprocess.py` to drop high‑missingness columns, impute missing values, encode categorical variables and label the target.
3. **Feature Engineering** – Execute `feature_engineering.py` to derive additional features such as ratios and temporal components.
4. **Model Training** – Train baseline and advanced models using `train_model.py`.  Save the trained models for later evaluation and prediction.
5. **Model Evaluation** – Compute performance metrics and create visualisations using `evaluate_model.py`.  Summarise results in Markdown reports.
6. **Insight Generation** – Review the EDA and model results to identify key drivers of default.  Document findings and recommendations.
7. **Deployment** – Launch the Streamlit app to allow interactive exploration and prediction.  Prepare the presentation to share with stakeholders.

---

This charter provides a roadmap for the project, aligning stakeholders on goals, scope and deliverables.  The following sections of the repository implement this plan in code, documentation and visualisations.