# Customer Churn Prediction Using XGBoost

## Abstract

Customer churn prediction is a critical task for telecommunications companies seeking to retain subscribers and minimize revenue loss. This study applies the XGBoost (Extreme Gradient Boosting) algorithm to the Telco Customer Churn dataset to build a binary classification model that predicts whether a customer will churn. The model achieved an accuracy of 79.56%, an F1-score of 0.5826, and an ROC-AUC of 0.8317. Feature importance analysis revealed that contract type is the single most influential predictor of churn, followed by online security and tech support availability.

## 1. Introduction

Customer churn, defined as the loss of clients or subscribers, represents one of the most significant challenges facing the telecommunications industry. Acquiring new customers costs five to seven times more than retaining existing ones, making churn prediction and prevention a high-priority business objective (Lalwani et al., 2022). The global telecommunications market experiences annual churn rates of 15--25%, translating to billions of dollars in lost revenue (Ahmad et al., 2019).

Machine learning has emerged as a powerful tool for churn prediction, enabling companies to proactively identify at-risk customers and implement targeted retention strategies. Among the various machine learning algorithms, gradient boosting methods -- particularly XGBoost -- have demonstrated superior performance in tabular classification tasks due to their ability to handle non-linear relationships, missing data, and class imbalance (Lalwani et al., 2022).

### 1.1 Literature Review

Recent research has extensively explored machine learning approaches for customer churn prediction. Ahmad et al. (2019) conducted a large-scale study comparing Decision Trees, Random Forest, Gradient Boosted Trees, and XGBoost on telecom data using Apache Spark, finding that XGBoost achieved the highest AUC (~0.84) among all tested models. Call detail record features such as total minutes of use and number of customer service calls were identified as the most important predictors.

Lalwani et al. (2022) performed a systematic comparison of Logistic Regression, SVM, Random Forest, AdaBoost, and XGBoost on the IBM Telco Customer Churn dataset. XGBoost outperformed all other classifiers with an F1-score of approximately 0.81 and an AUC of 0.85. Their feature importance analysis revealed that tenure, monthly charges, contract type, and internet service type are the top churn predictors.

Jain et al. (2020) evaluated Logistic Regression, LogitBoost, and Bayesian Network on telecom churn data, demonstrating that boosting-based methods consistently outperform standalone classifiers. Customer service call frequency and total day charges were identified as top features driving churn.

De Caigny et al. (2020) proposed a hybrid Logit Leaf Model and benchmarked it against XGBoost, Random Forest, and neural networks across 12 real-world churn datasets. XGBoost was consistently among the top-performing benchmarks, achieving high AUC across all datasets. The study demonstrated that hybrid and ensemble methods offer improved interpretability without sacrificing predictive performance.

Amin et al. (2020) proposed a data-certainty-based approach combined with ensemble classifiers, demonstrating that gradient boosting methods achieve AUC improvements of 3--5% when combined with data-certainty preprocessing. Contract duration, monthly charges, and tenure were validated as universal top-3 churn features across multiple telecom datasets.

Ullah et al. (2019) conducted a comprehensive comparison of eight classifiers on telecom churn datasets, finding that Gradient Boosted Trees and Random Forest were the two best-performing methods with AUC exceeding 0.90. The study emphasized the value of SHAP-based interpretability for deployment in business contexts.

Muneer et al. (2022) applied XGBoost, LightGBM, CatBoost, and other methods to banking customer churn prediction, achieving an F1-score of 0.87 and AUC of 0.88 with XGBoost, demonstrating the transferability of gradient boosting churn prediction methodology across domains.

## 2. Methods

### 2.1 Dataset

The Telco Customer Churn dataset, publicly available on Kaggle, contains 7,043 customer records with 19 features after preprocessing. Each record includes demographic information (gender, senior citizen status, partner, dependents), account information (tenure, contract type, payment method, monthly and total charges), and service subscriptions (phone service, internet service, online security, tech support, streaming services). The target variable is binary: whether the customer churned (Yes/No).

### 2.2 Data Preprocessing

The following preprocessing steps were applied:

1. **Missing value handling**: The `TotalCharges` column contained blank strings for customers with zero tenure; these were coerced to numeric and filled with zero.
2. **Feature removal**: The `customerID` column was dropped as it carries no predictive information.
3. **Target encoding**: The `Churn` column was mapped from categorical (Yes/No) to binary (1/0).
4. **Categorical encoding**: All remaining categorical features were transformed using Label Encoding.

### 2.3 Train-Test Split

The dataset was split into 80% training (5,634 samples) and 20% testing (1,409 samples) sets using stratified sampling to preserve the class distribution in both sets.

### 2.4 Model

XGBoost (Extreme Gradient Boosting) was selected as the classification algorithm. XGBoost is an optimized implementation of gradient boosting that uses regularized objective functions, efficient handling of sparse data, and parallel tree construction (Chen & Guestrin, 2016). The following hyperparameters were used:

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 5 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| eval_metric | logloss |

### 2.5 Evaluation Metrics

The model was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC. A confusion matrix and per-class classification report were also generated.

## 3. Results

### 3.1 Classification Performance

The XGBoost model achieved the following results on the test set:

| Metric | Value |
|--------|-------|
| Accuracy | 0.7956 |
| Precision | 0.6361 |
| Recall | 0.5374 |
| F1-Score | 0.5826 |
| ROC-AUC | 0.8317 |

### 3.2 Confusion Matrix

|  | Predicted: No Churn | Predicted: Churn |
|--|---------------------|------------------|
| **Actual: No Churn** | 920 (TN) | 115 (FP) |
| **Actual: Churn** | 173 (FN) | 201 (TP) |

### 3.3 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn (0) | 0.8417 | 0.8889 | 0.8647 | 1,035 |
| Churn (1) | 0.6361 | 0.5374 | 0.5826 | 374 |
| **Weighted Avg** | **0.7871** | **0.7956** | **0.7898** | **1,409** |

### 3.4 Feature Importance

The top 10 most important features as determined by XGBoost's gain-based importance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Contract | 0.4131 |
| 2 | OnlineSecurity | 0.0742 |
| 3 | TechSupport | 0.0694 |
| 4 | InternetService | 0.0552 |
| 5 | tenure | 0.0365 |
| 6 | PaperlessBilling | 0.0320 |
| 7 | MonthlyCharges | 0.0297 |
| 8 | StreamingMovies | 0.0294 |
| 9 | PhoneService | 0.0276 |
| 10 | TotalCharges | 0.0269 |

## 4. Discussion

The XGBoost model achieved an ROC-AUC of 0.8317, indicating good discriminative ability between churners and non-churners. This result is consistent with findings reported in the literature, where XGBoost typically achieves AUC values of 0.83--0.85 on the Telco Customer Churn dataset (Ahmad et al., 2019; Lalwani et al., 2022).

The model demonstrates a clear trade-off between precision (0.6361) and recall (0.5374) for the churn class. The relatively lower recall indicates that the model misses approximately 46% of actual churners, which could be addressed through threshold adjustment, class weighting, or oversampling techniques such as SMOTE (Lalwani et al., 2022).

Feature importance analysis reveals that **contract type** is overwhelmingly the most important predictor of churn, with an importance score of 0.4131 -- more than five times that of the next most important feature. This finding is consistent across the literature: customers with month-to-month contracts are significantly more likely to churn compared to those with one-year or two-year contracts (Amin et al., 2020; Ullah et al., 2019). Service-related features such as online security and tech support rank second and third, suggesting that customers without these add-on services are more churn-prone.

The presence of tenure (rank 5) and monthly charges (rank 7) among the top features validates the established understanding that newer customers and those with higher bills are more likely to leave (Lalwani et al., 2022). This information can directly inform business strategies, such as offering promotional pricing to new customers during the first year and bundling security and support services with base plans.

A limitation of this study is the use of Label Encoding for categorical features, which imposes an ordinal relationship where none may exist. One-Hot Encoding or target encoding could potentially improve model performance. Additionally, the current implementation does not address class imbalance explicitly, which may explain the relatively low recall on the minority churn class.

## 5. Conclusion

This study demonstrates the effectiveness of XGBoost for customer churn prediction in the telecommunications industry. The model achieved an ROC-AUC of 0.8317, confirming XGBoost's suitability for this task as established in the literature. The most critical finding is the dominant role of contract type in predicting churn, suggesting that retention strategies should prioritize converting month-to-month customers to longer-term contracts. Future work could explore SHAP-based explanations for individual predictions, hyperparameter optimization via Bayesian search, and ensemble approaches combining XGBoost with other boosting methods such as LightGBM and CatBoost.

## References

1. Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. *Journal of Big Data*, 6(1), Article 28.
2. Amin, A., Al-Obeidat, F., Shah, B., Adnan, A., Loo, J., & Anwar, S. (2020). Customer churn prediction in telecommunication industry using data certainty. *Journal of Business Research*, 94, 290--301.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785--794.
4. De Caigny, A., Coussement, K., & De Bock, K. W. (2020). A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees. *European Journal of Operational Research*, 269(2), 760--772.
5. Jain, H., Khunteta, A., & Srivastava, S. (2020). Churn prediction in telecommunication using logistic regression and logit boost. *Procedia Computer Science*, 167, 101--112.
6. Lalwani, P., Mishra, M. K., Chadha, J. S., & Sethi, P. (2022). Customer churn prediction system: A machine learning approach. *Computing*, 104(2), 271--294.
7. Muneer, A., Ali, R. F., Alghamdi, A., Taib, S. M., Almaghthawi, A., & Ghaleb, E. A. A. (2022). Predicting customers churning in banking industry: A machine learning approach. *Indonesian Journal of Electrical Engineering and Computer Science*, 26(1), 539--549.
8. Ullah, I., Raza, B., Malik, A. K., Imran, M., Islam, S. U., & Kim, S. W. (2019). A churn prediction model using random forest: Analysis of machine learning techniques for churn prediction and factor identification in telecom sector. *IEEE Access*, 7, 60134--60149.
