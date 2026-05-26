# Bank Customer Churn Prediction — Insights & Conclusion

---

## Libraries Used

| Library | Reason for Use |
|---|---|
| `numpy` | Efficient numerical operations and array handling throughout the pipeline |
| `pandas` | Loading the CSV dataset, cleaning data, dropping irrelevant columns, renaming columns, and exploratory analysis |
| `matplotlib` | Base layer for rendering all plots and customizing figure sizes |
| `seaborn` | High-level statistical plots — countplots, histplots with KDE, boxplots, violinplots, heatmaps for EDA and model evaluation |
| `sklearn.preprocessing.LabelEncoder` | Converts categorical columns `Gender` and `Geography` into numeric values since ML models require numeric input |
| `sklearn.preprocessing.StandardScaler` | Scales `CreditScore`, `Balance`, and `EstimatedSalary` to a common range so no single feature dominates due to magnitude differences |
| `sklearn.model_selection.train_test_split` | Splits data into 70% training and 30% testing to evaluate model performance on unseen data |
| `sklearn.model_selection.GridSearchCV` | Exhaustively searches over hyperparameter combinations with 5-fold cross-validation to find the best model configuration |
| `sklearn.tree.DecisionTreeClassifier` | First model — simple, interpretable, rule-based tree; used as a baseline classifier |
| `sklearn.ensemble.RandomForestClassifier` | Second model — ensemble of multiple decision trees; more robust and less prone to overfitting |
| `sklearn.metrics.confusion_matrix` | Visualizes True Positives, False Positives, True Negatives, False Negatives to understand prediction errors |
| `sklearn.metrics.classification_report` | Gives per-class precision, recall, and F1-score for a detailed evaluation beyond just accuracy |
| `sklearn.metrics.accuracy_score` | Measures the percentage of correctly classified customers overall |
| `sklearn.metrics.mean_absolute_error` | Measures the average magnitude of prediction errors |
| `sklearn.metrics.r2_score` | Indicates how well the model explains the variance in churn outcomes |

---

## EDA Insights

### 1. Churn Distribution
- Only **20.4%** of customers churned while **79.6%** were retained.
- The dataset is imbalanced — the majority class is non-churn.

### 2. Gender
- More male customers exist in the dataset than female.
- However, **females churn at a higher rate than males**.
- The difference is not large enough to make gender a strong standalone predictor.

### 3. Age
- Most customers fall in the **30–40 age group**.
- Churn is highest among customers aged **40–50**.
- Customers aged **20–25 have the lowest churn count**.
- **Age is a significant predictor** — older adults are far more likely to churn than young adults.

### 4. Credit Score
- The median credit score of churned and non-churned customers is almost identical.
- The distribution shape (violin plot) is also similar for both groups.
- Some churned customers have low credit scores, but overall **credit score is a weak predictor of churn**.

### 5. Geography
- France has the most customers, followed by Spain and Germany.
- Despite having fewer customers, **Germany has the highest churn count** relative to its size.
- **German customers are significantly more likely to churn** than French or Spanish customers.

### 6. Tenure
- Most customers have a tenure between **1–9 years** with fairly equal distribution.
- Churn is highest among customers with **1 year of tenure**.
- Customers with **10+ years of tenure churn the least** — long-term customers are more loyal.

### 7. Balance
- A large number of customers have a **zero balance**, and this group contributes heavily to churn.
- Customers with a balance between **1,00,000–1,50,000** also show a notably high churn rate.

### 8. Number of Products
- Customers using **1 or 2 products** are the most common and show low churn relative to their size.
- Customers using **3 or 4 products churn at an extremely high rate** — the number of leaving customers exceeds those staying in these categories.
- **Number of products is a strong indicator of churn.**

### 9. Credit Card Ownership
- Nearly **70% of customers hold a credit card**.
- Interestingly, customers **with** a credit card have a higher absolute churn count than those without.

### 10. Active Membership
- **Inactive members churn significantly more** than active members.
- Active members are more satisfied with bank services and hence more loyal.
- This makes **active membership status a key churn indicator**.

### 11. Estimated Salary
- No definite pattern exists between salary and churn.
- Churned and non-churned customers are spread across all salary ranges.
- **Estimated salary is not a useful predictor of churn.**

### 12. Correlation Matrix
- No significant correlation exists between any two independent variables.
- No multicollinearity issues — all features can be used independently in model training.

---

## Model Insights

### Decision Tree Classifier
- Hyperparameter tuned using `GridSearchCV` (5-fold CV, scoring = ROC-AUC)
- Best parameters: `criterion='gini'`, `max_depth=6`, `min_samples_leaf=10`, `random_state=42`
- `max_depth=6` was used to prevent overfitting by limiting tree complexity.
- The confusion matrix shows good True Positive performance but some misclassifications on the minority churn class due to class imbalance.

### Random Forest Classifier
- Hyperparameter tuned using `GridSearchCV` (5-fold CV, scoring = ROC-AUC)
- Best parameters: `criterion='entropy'`, `max_depth=10`, `min_samples_leaf=8`, `random_state=0`
- Being an ensemble model, it generalizes better than a single decision tree.
- Achieves better accuracy and precision than the Decision Tree.

---

## Conclusion

### Key Churn Drivers
From the EDA, the following features are the most influential in predicting customer churn:

1. **Age** — Customers aged 40–50 are most at risk; younger customers rarely churn.
2. **Geography** — German customers churn at a disproportionately high rate.
3. **Number of Products** — Customers using 3 or 4 products are overwhelmingly likely to churn.
4. **Active Membership** — Inactive members are far more likely to leave the bank.
5. **Tenure** — New customers (1 year) are at higher risk; long-term customers are loyal.
6. **Balance** — Zero-balance customers and those in the 1,00,000–1,50,000 range churn more.

### Weak or Non-Predictive Features
- **Credit Score** — Minimal difference between churned and retained customers.
- **Estimated Salary** — No consistent pattern observed.
- **Gender** — Slight difference, but not a strong standalone predictor.

### Model Comparison

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | ~86% | ~87% |
| Precision (Churn class) | Moderate | Higher |
| Recall (Churn class) | Moderate | Moderate |
| MAE | Low | Lower |
| Overfitting Risk | Moderate | Low |

### Final Recommendation
The **Random Forest Classifier is the better model** for this task. It outperforms the Decision Tree in accuracy and precision, and being an ensemble method, it is more robust to noise and overfitting.

Banks can deploy this model to score customers periodically and flag high-risk churners. Retention efforts should be focused on:
- **Inactive members** → re-engagement campaigns
- **German customers** → region-specific service improvement
- **Customers using 3–4 products** → review product bundling strategy
- **New customers (tenure < 2 years)** → early loyalty programs
