# Model Evaluation

This document presents a concise evaluation of the trained models on the Lending Club dataset.  The evaluation metrics and plots help assess how well each model discriminates between defaulting and fully paid loans.  The results below are based on the same simulated test set used in the model comparison report.

## Metrics Overview

The following table summarises the evaluation metrics for each model.  See the **Model Comparison** report for a detailed discussion of model characteristics and performance.

| Model | Accuracy | Precision | Recall | F1‑Score | ROC–AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.825** | **0.844** | **0.963** | **0.900** | **0.776** |
| **Random Forest** | 0.818 | 0.839 | 0.961 | 0.896 | 0.746 |
| **XGBoost** | 0.812 | 0.844 | 0.944 | 0.891 | 0.736 |
| **LightGBM** | 0.817 | 0.849 | 0.942 | 0.893 | 0.733 |

### Interpretation

* **Accuracy:** All models achieve accuracies around 0.81–0.83.  Accuracy alone can be misleading because the dataset is imbalanced; therefore we examine recall and precision.
* **Recall:** Recall measures the proportion of actual defaulters correctly identified.  Each model attains recall above 0.94, indicating that the models are effective at catching most defaults.
* **Precision:** Precision reflects the proportion of predicted defaulters that actually default.  Values around 0.84–0.85 show a moderate false‑positive rate; lenders may tune the probability threshold to trade precision against recall based on risk appetite.
* **F1‑Score:** The harmonic mean of precision and recall, F1‑score summarises model performance on imbalanced data.  Logistic regression has the highest F1‑score (0.900) in this simulation, closely followed by the ensemble models.
* **ROC–AUC:** Area under the Receiver Operating Characteristic curve quantifies overall discriminatory power across all probability thresholds.  Logistic regression again leads with an AUC of 0.776, with tree‑based models achieving slightly lower but still respectable values.

## Confusion Matrices

For each model, a confusion matrix visualises true positives, false positives, true negatives and false negatives.  In this project, high recall leads to relatively few false negatives (missed defaulters), while the number of false positives (non‑defaulters incorrectly classified as defaulters) varies by model.  These matrices can be found in the `visuals/` directory of the repository.

## ROC Curves

The ROC curves illustrate the trade‑off between true positive rate (recall) and false positive rate at various classification thresholds.  A model with an ROC curve closer to the top‑left corner of the plot is better.  In our results, logistic regression and the tree‑based models have similar ROC curves, with logistic regression slightly superior.  Plots of the ROC curves are also stored in the `visuals/` directory.

## Conclusion

The evaluation shows that all models perform strongly on the simulated Lending Club test set, with high recall and competitive precision.  Logistic regression provides a robust baseline, while tree‑based methods offer comparable performance and may surpass the baseline when tuned on the real Lending Club data.  Ultimately, model selection should consider both performance metrics and business requirements such as interpretability, prediction latency and regulatory compliance.