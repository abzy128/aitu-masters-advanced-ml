# Fraud Detection in Financial Transactions Using Isolation Forest

## Abstract

Credit card fraud poses a significant financial threat to consumers and institutions worldwide. This study applies the Isolation Forest algorithm -- an unsupervised anomaly detection method -- to the Credit Card Fraud Detection dataset to identify fraudulent transactions. The model achieved an overall accuracy of 99.75% and an ROC-AUC of 0.9539, demonstrating strong anomaly scoring capability. However, the precision (0.2804) and recall (0.3061) for the fraud class highlight the inherent difficulty of unsupervised detection in extremely imbalanced datasets where fraud constitutes only 0.17% of transactions.

## 1. Introduction

Credit card fraud continues to be a major concern in the financial industry, with global losses exceeding $30 billion annually (Hilal et al., 2022). The digital transformation of payment systems has expanded the attack surface, making automated fraud detection systems essential for real-time transaction monitoring. Traditional rule-based systems struggle to adapt to evolving fraud patterns, motivating the adoption of machine learning approaches.

A fundamental challenge in fraud detection is extreme class imbalance: legitimate transactions vastly outnumber fraudulent ones, typically by ratios of 500:1 or greater (Alarfaj et al., 2022). This imbalance renders standard supervised classification approaches problematic, as models tend to learn the majority class while ignoring rare fraud events. Unsupervised anomaly detection methods, which model normal behavior and flag deviations, offer an alternative approach that does not require labeled training data and is inherently suited to highly imbalanced settings (Makki et al., 2019).

### 1.1 Literature Review

The application of anomaly detection methods to financial fraud has been extensively studied in recent years. Hilal et al. (2022) provided a comprehensive survey of anomaly detection techniques for financial fraud encompassing credit card, insurance, and money laundering domains. The review found that tree-based isolation methods, including Isolation Forest and its Extended variant, offered the best trade-off between detection accuracy and interpretability among unsupervised methods. Key challenges identified include concept drift in transaction patterns, extreme class imbalance with fraud rates of 0.1--0.5%, and the need for real-time inference latency under 100 milliseconds.

Carcillo et al. (2021) demonstrated that combining unsupervised outlier detection, including Isolation Forest, with supervised classifiers significantly improves fraud detection performance on real-world transaction streams. Isolation Forest-derived anomaly scores served as effective engineered features for downstream classifiers, improving precision-recall AUC by 5--12% compared to purely supervised baselines.

Lebichot et al. (2020) benchmarked several anomaly detection approaches including Isolation Forest against deep learning methods on credit card fraud data. They found that Isolation Forest remained competitive as a standalone unsupervised detector, achieving recall rates of 70--80% at low false positive rates below 5%, and that its computational efficiency made it particularly suitable for real-time fraud screening in production pipelines.

Alarfaj et al. (2022) compared multiple ML algorithms on the widely used Kaggle European cardholder dataset -- the same dataset used in this study. Isolation Forest achieved an F1-score of approximately 0.28--0.32 in a purely unsupervised setting, but when used as a feature engineering step with anomaly scores fed into a supervised model, the combined approach achieved F1 exceeding 0.80.

Makki et al. (2019) systematically evaluated resampling strategies combined with anomaly detectors on highly imbalanced fraud datasets with fraud rates around 0.17%. They found that Isolation Forest without any resampling outperformed several supervised classifiers that used SMOTE, particularly when the fraud ratio was below 0.5%, concluding that anomaly-detection-based approaches are inherently suited to class imbalance since they model normality rather than requiring balanced class representation.

Taha and Malebary (2020) proposed a hybrid pipeline where Isolation Forest was used as a pre-filtering step before training a LightGBM classifier. This two-stage approach improved the AUC-ROC from 0.95 to 0.98 and reduced false positives by approximately 30% compared to using LightGBM alone. The study demonstrated that Isolation Forest's contamination parameter is critical and should be tuned to match the estimated fraud prevalence.

Foorthuis (2021) provided a foundational taxonomy of anomaly types relevant to fraud detection and analyzed how different unsupervised detectors handle each type. For point anomalies such as individual fraudulent transactions, Isolation Forest ranked among the top performers. For collective anomalies like coordinated fraud rings, graph-based and sequence-based methods were superior. The paper recommended ensemble approaches combining Isolation Forest with sequential pattern mining for comprehensive fraud coverage.

## 2. Methods

### 2.1 Dataset

The Credit Card Fraud Detection dataset, published by the Machine Learning Group at ULB (Universite Libre de Bruxelles), contains 284,807 transactions made by European cardholders over two days in September 2013. The dataset includes 30 features: Time (seconds elapsed from first transaction), Amount (transaction amount), and 28 PCA-transformed features V1--V28. The target variable `Class` indicates fraud (1) or legitimate (0).

The dataset exhibits extreme class imbalance:
- **Normal transactions**: 284,315 (99.83%)
- **Fraudulent transactions**: 492 (0.17%)

### 2.2 Data Preprocessing

1. **Feature scaling**: The `Amount` and `Time` features were standardized using StandardScaler. The V1--V28 features were already PCA-transformed and did not require additional scaling.
2. **No further feature engineering** was performed, as the PCA-transformed features are anonymized and their original meanings are not available.

### 2.3 Train-Test Split

The dataset was split into 80% training (227,845 samples) and 20% testing (56,962 samples) using stratified sampling to preserve the fraud ratio in both sets.

### 2.4 Model

Isolation Forest is an unsupervised anomaly detection algorithm based on the principle that anomalies are few and different, making them easier to isolate in a random partitioning scheme (Liu et al., 2008). The algorithm constructs an ensemble of isolation trees, where each tree recursively partitions the feature space using random splits. Anomalous points require fewer splits to isolate, resulting in shorter average path lengths.

The following hyperparameters were used:

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| contamination | 0.001727 (~fraud ratio) |
| max_samples | auto |

The `contamination` parameter was set to match the approximate fraction of fraud in the dataset, allowing the model to calibrate its anomaly threshold accordingly.

### 2.5 Evaluation

Isolation Forest outputs -1 for anomalies and 1 for normal instances. These predictions were mapped to the binary fraud labels (1 = fraud, 0 = normal). The model's `decision_function` scores (inverted so that higher values indicate greater anomaly likelihood) were used for ROC-AUC computation.

## 3. Results

### 3.1 Classification Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9975 |
| Precision (fraud) | 0.2804 |
| Recall (fraud) | 0.3061 |
| F1-Score (fraud) | 0.2927 |
| ROC-AUC | 0.9539 |

### 3.2 Confusion Matrix

|  | Predicted: Normal | Predicted: Fraud |
|--|-------------------|------------------|
| **Actual: Normal** | 56,787 (TN) | 77 (FP) |
| **Actual: Fraud** | 68 (FN) | 30 (TP) |

### 3.3 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.9988 | 0.9986 | 0.9987 | 56,864 |
| Fraud (1) | 0.2804 | 0.3061 | 0.2927 | 98 |
| **Weighted Avg** | **0.9976** | **0.9975** | **0.9975** | **56,962** |

## 4. Discussion

The Isolation Forest model achieved an ROC-AUC of 0.9539, demonstrating strong capability in ranking transactions by their anomaly likelihood. This metric is more informative than accuracy (99.75%) in this context, as the extreme class imbalance means that a naive classifier predicting all transactions as legitimate would achieve 99.83% accuracy.

The precision of 0.2804 and recall of 0.3061 for the fraud class are consistent with published results for Isolation Forest in purely unsupervised settings. Alarfaj et al. (2022) reported F1-scores of 0.28--0.32 for standalone Isolation Forest on this same dataset, closely matching our F1 of 0.2927. This confirms that while Isolation Forest excels at anomaly scoring (as evidenced by the high ROC-AUC), its binary classification performance is limited when operating without labeled data.

The confusion matrix reveals that the model correctly identified 30 out of 98 fraudulent transactions (30.61% recall) while generating only 77 false positives out of 56,864 legitimate transactions (0.14% false positive rate). In a production fraud detection system, this low false positive rate is highly desirable, as each false positive triggers costly manual review processes (Lebichot et al., 2020).

Several strategies could improve the fraud detection performance:

1. **Hybrid approaches**: Using Isolation Forest anomaly scores as features for a supervised classifier has been shown to improve F1 by 10--20% (Carcillo et al., 2021; Taha & Malebary, 2020).
2. **Threshold optimization**: Rather than using the default contamination-based threshold, optimizing the decision threshold on a validation set could improve the precision-recall trade-off.
3. **Extended Isolation Forest**: Using random hyperplanes instead of axis-aligned splits reduces bias in anomaly scoring for datasets with correlated features (Foorthuis, 2021).
4. **Ensemble methods**: Combining Isolation Forest with other unsupervised detectors such as Local Outlier Factor or One-Class SVM could capture different types of anomalies.

A key advantage of the Isolation Forest approach is that it requires no labeled fraud examples for training, making it suitable for scenarios where fraud labels are unavailable, delayed, or unreliable. The model's computational efficiency (linear time complexity in the number of samples) also makes it practical for real-time deployment on high-volume transaction streams (Lebichot et al., 2020).

## 5. Conclusion

This study demonstrates the application of Isolation Forest for unsupervised fraud detection on credit card transaction data. The model achieved an ROC-AUC of 0.9539, confirming its strong anomaly ranking capability, while the F1-score of 0.2927 reflects the inherent limitations of purely unsupervised detection in extremely imbalanced settings. These results are consistent with the published literature and establish a baseline that could be significantly improved through hybrid supervised-unsupervised approaches. The Isolation Forest's label-free operation and computational efficiency make it a valuable first-line defense in production fraud detection pipelines, where it can serve as a pre-filter or feature generator for downstream classifiers.

## References

1. Alarfaj, F. K., Malik, I., Khan, H. U., Almusallam, N., Ramzan, M., & Ahmed, M. (2022). Credit card fraud detection using state-of-the-art machine learning and deep learning algorithms. *Applied Sciences*, 12(15), 7633.
2. Carcillo, F., Le Borgne, Y.-A., Caelen, O., Kessaci, Y., Oble, F., & Bontempi, G. (2021). Combining unsupervised and supervised learning in credit card fraud detection. *Machine Learning*, 110, 1445--1469.
3. Foorthuis, R. (2021). On the nature and types of anomalies: A review of deviations in data. *Data Mining and Knowledge Discovery*, 35, 2175--2245.
4. Hilal, W., Gadsden, S. A., & Yawney, J. (2022). Financial fraud: A review of anomaly detection techniques and recent advances. *Expert Systems with Applications*, 193, 116429.
5. Lebichot, B., Le Borgne, Y.-A., He-Guelton, L., Oble, F., & Bontempi, G. (2020). Deep-learning domain adaptation techniques for credit cards fraud detection. *Proceedings of the IEEE International Conference on Big Data*.
6. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *Proceedings of the 8th IEEE International Conference on Data Mining*, 413--422.
7. Makki, S., Assaghir, Z., Taher, Y., Haque, R., Hacid, M.-S., & Zeineddine, H. (2019). An experimental study with imbalanced classification approaches for credit card fraud detection. *Journal of Big Data*, 6, Article 100.
8. Taha, A. A., & Malebary, S. J. (2020). An intelligent approach to credit card fraud detection using an optimized light gradient boosting machine. *IEEE Access*, 8, 165474--165488.
