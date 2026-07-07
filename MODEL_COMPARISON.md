# Model Comparison

This document compares the baseline and advanced machine‑learning models trained on the engineered Lending Club dataset.  Because the raw Lending Club dataset contains a large number of features and significant missingness, we first preprocess it by dropping high‑missing columns, imputing values, encoding categorical variables and engineering features.  We then split the data into training and test sets and evaluate each model on the same test set.

## Models Evaluated

| Model | Description |
|---|---|
| **Logistic Regression** | A linear baseline classifier that models the log‑odds of default as a weighted sum of input features.  It is interpretable and handles sparse one‑hot encoded features well. |
| **Random Forest** | An ensemble of decision trees trained on bootstrap samples with feature bagging.  Captures nonlinear relationships and interactions without requiring feature scaling. |
| **XGBoost** | Gradient boosting algorithm that builds trees sequentially to correct previous errors.  Often provides strong performance on tabular datasets. |
| **LightGBM** | A gradient boosting implementation optimised for efficiency and capable of handling categorical features through built‑in support for histogram‑based splits. |

## Evaluation Metrics

We evaluate models using **accuracy**, **precision**, **recall**, **F1‑score** and **ROC–AUC** on the held‑out test set.  In credit‑risk prediction, **recall** (how many true defaulters are identified) and **precision** (how many predicted defaulters are actually defaulters) are especially important.  **F1‑score** provides a balanced measure of precision and recall, while **ROC–AUC** summarises discrimination across all thresholds.

## Performance Summary

The table below shows the performance of each model on the test set (values are rounded to three decimals).  These numbers are based on a simulated dataset with similar characteristics to the real Lending Club data and should be viewed as illustrative rather than definitive.

| Model | Accuracy | Precision | Recall | F1‑Score | ROC–AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.825** | **0.844** | **0.963** | **0.900** | **0.776** |
| **Random Forest** | 0.818 | 0.839 | 0.961 | 0.896 | 0.746 |
| **XGBoost** | 0.812 | 0.844 | 0.944 | 0.891 | 0.736 |
| **LightGBM** | 0.817 | 0.849 | 0.942 | 0.893 | 0.733 |

## Discussion

* **Baseline vs. advanced models:** In this simulation the baseline logistic regression model performs surprisingly well, achieving the highest F1‑score and ROC–AUC.  This underscores that a well‑regularised linear model can be strong when feature engineering and preprocessing are done carefully.  However, the advanced models (Random Forest, XGBoost, LightGBM) perform comparably and may generalise better on more complex real‑world data because they capture nonlinear interactions.
* **Recall:** All models achieve high recall (> 0.94) because the engineered features, such as debt‑to‑income ratio and credit grade, provide strong signals.  High recall means the models are effective at identifying defaulters.
* **Precision:** Precision values around 0.84–0.85 indicate that some predicted defaulters are false positives.  In a lending context this translates to lost opportunity cost, so threshold tuning can be used to balance precision and recall according to business risk tolerance.
* **ROC–AUC:** Logistic regression has the highest area under the ROC curve (0.776), suggesting it discriminates slightly better between defaulters and non‑defaulters across all thresholds.  Tree‑based models still perform well but show marginally lower AUC in this simulation.

## Conclusion

While the simulated results suggest that logistic regression provides a strong baseline, in practice the choice between linear and ensemble models depends on the real data and business objectives.  Tree‑based models often excel when there are complex interactions among variables, and gradient boosting methods like XGBoost or LightGBM may achieve higher performance after hyper‑parameter tuning.  For deployment, we recommend experimenting with these advanced models, tuning their parameters via cross‑validation and comparing metrics such as F1‑score and ROC–AUC on a held‑out validation set.