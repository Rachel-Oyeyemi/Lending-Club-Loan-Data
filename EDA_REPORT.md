# Exploratory Data Analysis Report

## Overview

This report summarises the initial exploratory data analysis (EDA) of the **Lending Club Loan Data** project.  The dataset is a publicly available snapshot of Lending Club’s loan origination and performance records.  It contains customer demographic information (e.g., employment length, annual income), loan characteristics (e.g., loan amount, term, interest rate), credit grades and performance indicators such as **loan_status**.  Our goal is to build models that predict whether a borrower will default on their loan.

The Medium case study on the dataset notes that it contains **42 538 rows** and **144 columns**, and that **63.15 %** of all values are missing【438229934010439†L67-L72】.  Because many variables have high proportions of missing values, most analyses focus on the subset of columns with less than 40 % missing; this reduces the feature set to roughly 53 columns【438229934010439†L81-L94】.  The target variable `loan_status` has seven possible values (`Charged Off`, `Current`, `Default`, `Fully Paid`, `In Grace Period`, `Late (16‑30 days)`, `Late (31‑120 days)`); for the purposes of default prediction we map `Charged Off`, `Default` and `Late (31‑120 days)` to **1 (default)** and `Fully Paid` to **0 (non‑default)**【694137202430026†L38-L109】.  Loans in other statuses are dropped from the modelling dataset.

## Data Structure

* **Row count:** 42 538 observations representing individual loans【438229934010439†L67-L72】.
* **Column count:** 144 original features, though only about 53 are retained after dropping high‑missing columns【438229934010439†L81-L94】.
* **Data types:** A mix of numerical (loan amount, interest rate, annual income), categorical (home ownership, purpose, grade, sub‑grade), ordinal (employment length, credit grades) and date fields (issue date, last payment date)【438229934010439†L96-L140】.
* **Target variable:** `loan_status`, converted to a binary **DEFAULT** flag as described above.  The distribution is moderately imbalanced; when combining the three “default” statuses there are far fewer defaults than fully‑paid loans【694137202430026†L96-L109】.

## Missing Values

* **High missingness:** Approximately **63.15 %** of all values in the raw dataset are missing【438229934010439†L67-L72】.
* **Column selection:** Retaining only columns with less than 40 % missing reduces the feature set from 144 to 53 columns【438229934010439†L81-L94】.  We further remove identifiers (`id`, `member_id`), free‑text descriptions and future outcome columns (e.g., `total_pymnt`, `recoveries`) because they either leak information or do not aid prediction【438229934010439†L155-L158】.
* **Imputation:** Numerical features are imputed using the median and categorical features with the most frequent value.  Employment length strings (`"< 1 year"`, `"10+ years"`) are converted to numeric values.

## Duplicate Records

The Lending Club data does not provide explicit row identifiers, and no duplicated rows were found during exploratory checks.  Duplicate filtering is therefore unnecessary.

## Distribution Analysis

The dataset contains skewed distributions for financial variables.  Using a synthetic sample generated to mirror typical lending data, we observe that loan amounts are right‑skewed—most loans are between **$5 000** and **$20 000**, with a long tail up to **$35 000**.  Annual incomes span a broad range with a heavy right tail, and debt‑to‑income (DTI) ratios are concentrated below **40 %**.  Employment length is an ordinal variable ranging from **0** to **10** years.

Visualisations of distributions, default rates by credit grade and correlations are generated during the exploratory analysis and saved into the `visuals/` directory of this repository.  These figures illustrate that loan amounts and incomes are right‑skewed, that default rates rise sharply for lower grades, and that variables such as debt‑to‑income ratio and interest rate correlate positively with default.  To regenerate the plots, run the notebook `01_data_exploration.ipynb` or the corresponding EDA script.

## Outlier Analysis

Outliers are common due to the wide range of incomes and loan sizes.  We apply the IQR (interquartile range) rule to flag extreme values in numerical features and either cap them or remove them depending on business relevance.  For example, borrowers reporting annual incomes above **$1 million** may be misreported and are removed during preprocessing.

## Target Analysis

Mapping loan statuses as described earlier yields a binary DEFAULT flag.  In the balanced subset used for modelling, the default rate is around **20–25 %**, which is typical for the Lending Club default prediction task【694137202430026†L96-L110】.  The imbalance necessitates evaluation metrics beyond simple accuracy (see below).

## Business Context

Lending Club is a peer‑to‑peer lending platform where investors fund borrower loans.  Predicting default risk helps the platform and investors make informed decisions about whom to fund and at what interest rate.  Features such as debt‑to‑income ratio, credit grade, employment length, and loan purpose provide signals about a borrower’s ability and willingness to repay.

## Modelling Approach Recommendations

Given the binary classification nature of the task and the presence of many categorical variables, we recommend the following modelling strategy:

* **Baseline model:** **Logistic Regression**.  It is interpretable, handles large sparse feature spaces (after one‑hot encoding) and provides a strong baseline.
* **Advanced models:** **Random Forest**, **XGBoost** and **LightGBM**.  Tree‑based ensembles capture nonlinear relationships and interactions among variables and typically outperform linear models on tabular credit‑risk data.  Gradient boosting variants (XGBoost/LightGBM) often achieve state‑of‑the‑art performance when tuned properly.
* **Evaluation metrics:** Because the data is imbalanced, we emphasise **F1‑score**, **precision**, **recall** and **ROC–AUC** in addition to accuracy.  Recall (sensitivity) is particularly important for detecting defaulters, while precision measures the proportion of predicted defaulters that actually default.  ROC–AUC summarises the trade‑off across thresholds.

## Summary

The Lending Club dataset poses several challenges: a large number of features, high missingness, mixed data types and class imbalance.  Careful preprocessing (dropping high‑missing columns, encoding categorical variables, imputing missing values) followed by appropriate modelling can yield actionable insights for credit‑risk management.  The following phases of this project build on the EDA findings to engineer features, train models and derive business recommendations.