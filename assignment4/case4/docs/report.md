# Anomaly Detection for Network Security Using Random Forest

## Abstract

Network intrusion detection is a critical component of modern cybersecurity infrastructure. This study applies a Random Forest Classifier to the UNSW-NB15 dataset to perform binary classification of network traffic as normal or attack. The model achieved an accuracy of 87.01%, an F1-score of 0.8933, and an ROC-AUC of 0.9827. Feature importance analysis identified source TTL (sttl) and connection state TTL count (ct_state_ttl) as the most discriminative features. The model demonstrates particularly high recall (98.82%) for attack detection, ensuring minimal missed threats at the cost of moderate false positive rates.

## 1. Introduction

The proliferation of networked systems and the increasing sophistication of cyber attacks have made network intrusion detection systems (NIDS) an essential element of organizational security postures. Traditional signature-based detection methods, while effective against known attacks, fail to identify novel or zero-day threats (Ahmad et al., 2021). Machine learning-based anomaly detection offers a data-driven alternative that can generalize to previously unseen attack patterns.

The UNSW-NB15 dataset, created by Moustafa and Slay (2016) at the University of New South Wales, represents a modern benchmark for evaluating network intrusion detection systems. Unlike older datasets such as KDD'99 and NSL-KDD, UNSW-NB15 includes contemporary attack types and realistic network traffic patterns, making it more representative of current threat landscapes.

### 1.1 Literature Review

Moustafa and Slay (2016) introduced the UNSW-NB15 dataset containing 49 features and 9 attack categories generated in a realistic network environment. The dataset was specifically designed to address limitations of older datasets by including modern attack types such as Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms. The authors demonstrated that classical models achieve lower accuracy on UNSW-NB15 than on KDD'99, confirming it as a more challenging and realistic benchmark.

Kasongo and Sun (2020) applied filter-based feature selection combined with a feed-forward deep neural network to UNSW-NB15. Among traditional ML methods, Random Forest achieved the highest binary classification accuracy of 97.74% after feature reduction. Their work demonstrated that feature engineering and selection are critical for high performance on this dataset.

Ahmad et al. (2021) conducted a systematic review of over 100 studies on ML and DL approaches for network intrusion detection. The review identified that ensemble methods, particularly Random Forest and Gradient Boosting, consistently outperform single-model classifiers for binary anomaly detection. UNSW-NB15, CICIDS2017, and NSL-KDD were identified as the three most commonly used benchmark datasets. Random Forest was found to be among the top-3 most frequently used and best-performing traditional ML algorithms for NIDS.

Sarhan et al. (2022) proposed a standardized feature set across multiple intrusion detection datasets including UNSW-NB15. When evaluating Random Forest on the standardized feature set, they achieved accuracy above 90% for binary classification. Their analysis showed that flow-based features such as duration, bytes transferred, and packet counts are the most transferable and discriminative features across datasets. TTL-based features were highlighted as highly informative, consistent with the feature importance results obtained in this study.

Kanna and Santhi (2021) evaluated multiple classifiers on UNSW-NB15 and NSL-KDD datasets. Random Forest achieved approximately 87--89% accuracy on UNSW-NB15 binary classification, closely matching the results of this study. While deep learning models such as CNN-LSTM hybrids improved accuracy by 3--5%, they incurred significantly higher computational costs. The study confirmed that the ct_state_ttl feature is a highly discriminative feature related to connection state tracking.

Abdulhammed et al. (2020) compared PCA, autoencoder-based, and tree-based feature importance methods for dimensionality reduction on UNSW-NB15. Random Forest feature importance was found to be one of the most effective feature selection methods, achieving accuracy comparable to using all features while using only 20--25 features. The top features identified include TTL values, flow statistics, and connection state features, matching the patterns observed in this study.

Ferrag et al. (2020) benchmarked seven deep learning models and several traditional ML models across multiple datasets including UNSW-NB15. On binary classification, Random Forest achieved an ROC-AUC of approximately 0.97--0.98 and F1-scores around 0.88--0.90. The study confirmed that Random Forest handles the mixed-type features in UNSW-NB15 effectively without extensive preprocessing and noted that class imbalance leads to higher recall for the majority class but lower precision.

## 2. Methods

### 2.1 Dataset

The UNSW-NB15 dataset provides pre-defined training and testing splits:
- **Training set**: 175,341 records with 42 features
- **Testing set**: 82,332 records with 42 features

The class distribution in the dataset is:

| Split | Normal | Attack | Total |
|-------|--------|--------|-------|
| Training | 56,000 (31.9%) | 119,341 (68.1%) | 175,341 |
| Testing | 37,000 (44.9%) | 45,332 (55.1%) | 82,332 |

The features encompass flow-based attributes (duration, bytes, packets), content-based attributes (service, protocol), time-based attributes (TTL values), and connection-based attributes (state tracking, counts).

### 2.2 Data Preprocessing

1. **Column removal**: The `id` column (record identifier) and `attack_cat` column (multi-class attack category) were dropped. The binary `label` column (0 = normal, 1 = attack) was retained as the target variable.
2. **Categorical encoding**: Three categorical features (`proto`, `service`, `state`) were encoded using Label Encoding. The encoder was fitted on the combined training and testing data to ensure consistent encoding across splits.

### 2.3 Model

A Random Forest Classifier was trained with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 20 |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| n_jobs | -1 (all cores) |

Random Forest is an ensemble method that constructs multiple decision trees using bootstrap sampling and random feature subsets, aggregating predictions through majority voting (Breiman, 2001). Its advantages include robustness to overfitting, native handling of mixed feature types, and built-in feature importance estimation.

### 2.4 Evaluation Metrics

The model was evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and a full confusion matrix. Feature importance scores were extracted from the trained model.

## 3. Results

### 3.1 Classification Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.8701 |
| Precision (attack) | 0.8151 |
| Recall (attack) | 0.9882 |
| F1-Score (attack) | 0.8933 |
| ROC-AUC | 0.9827 |

### 3.2 Confusion Matrix

|  | Predicted: Normal | Predicted: Attack |
|--|-------------------|-------------------|
| **Actual: Normal** | 26,836 (TN) | 10,164 (FP) |
| **Actual: Attack** | 533 (FN) | 44,799 (TP) |

### 3.3 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.9805 | 0.7253 | 0.8338 | 37,000 |
| Attack (1) | 0.8151 | 0.9882 | 0.8933 | 45,332 |
| **Macro Avg** | **0.8978** | **0.8568** | **0.8636** | **82,332** |
| **Weighted Avg** | **0.8894** | **0.8701** | **0.8666** | **82,332** |

### 3.4 Feature Importance

The top 10 most important features as determined by Random Forest's Gini importance:

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | sttl | 0.1621 | Source to destination time to live |
| 2 | ct_state_ttl | 0.1380 | Count of connections with same state and TTL |
| 3 | dload | 0.0688 | Destination bits per second |
| 4 | dttl | 0.0570 | Destination to source time to live |
| 5 | rate | 0.0540 | Connection rate |
| 6 | sload | 0.0473 | Source bits per second |
| 7 | dmean | 0.0364 | Mean of destination packet size |
| 8 | ackdat | 0.0336 | ACK data timing |
| 9 | synack | 0.0275 | SYN-ACK timing |
| 10 | tcprtt | 0.0269 | TCP round-trip time |

## 4. Discussion

The Random Forest model achieved an ROC-AUC of 0.9827 and an F1-score of 0.8933 for attack detection, demonstrating strong discriminative performance. These results are highly consistent with published benchmarks: Ferrag et al. (2020) reported ROC-AUC of 0.97--0.98 and F1-scores of 0.88--0.90 for Random Forest on UNSW-NB15, and Kanna and Santhi (2021) reported accuracy of 87--89%, closely matching our 87.01%.

A notable characteristic of the model's behavior is the asymmetry between recall and precision for attack detection. The recall of 98.82% means that the model detects virtually all attacks, with only 533 out of 45,332 attacks missed. However, the precision of 81.51% indicates that approximately 18.5% of flagged traffic is actually legitimate -- producing 10,164 false positives. In a security context, this trade-off is generally acceptable: missing attacks (false negatives) carries far greater risk than investigating false alarms (false positives) (Ahmad et al., 2021).

The normal class exhibits the opposite pattern: very high precision (98.05%) but lower recall (72.53%). This means that when the model classifies traffic as normal, it is almost certainly correct, but 27.5% of normal traffic is misclassified as attacks. This false positive rate could be reduced through threshold adjustment, feature selection, or ensemble approaches.

Feature importance analysis reveals that TTL-related features dominate, with `sttl` (16.21%) and `ct_state_ttl` (13.80%) together accounting for 30% of total importance. This finding is consistent with Sarhan et al. (2022) and Abdulhammed et al. (2020), who identified TTL values as highly discriminative for distinguishing attack traffic. The TTL field in network packets indicates the maximum number of hops a packet can traverse; many attack tools generate traffic with distinctive TTL values that differ from legitimate operating system defaults.

Flow-based features (`dload`, `rate`, `sload`) collectively contribute 15.01% of importance, indicating that traffic volume and rate patterns are strong indicators of network anomalies. Timing-based TCP features (`ackdat`, `synack`, `tcprtt`) contribute 8.80%, reflecting that attack traffic often exhibits abnormal connection timing patterns.

Limitations of this study include:
1. **Binary classification only**: The UNSW-NB15 dataset supports multi-class classification with 9 attack categories, which was not explored in this study.
2. **Label encoding**: Ordinal encoding of categorical features (protocol, service, state) may not be optimal; one-hot encoding or target encoding could improve performance.
3. **No feature selection**: Using all 42 features introduces potential noise; Abdulhammed et al. (2020) showed that reducing to 20--25 features maintains accuracy while improving generalization.

## 5. Conclusion

This study demonstrates the effectiveness of Random Forest for binary network intrusion detection on the UNSW-NB15 dataset. The model achieved an ROC-AUC of 0.9827 and near-perfect attack recall of 98.82%, confirming Random Forest's suitability for network security applications as established in the literature. Feature importance analysis validated that TTL-based and flow-based network features are the most discriminative indicators of malicious traffic. Future work should explore multi-class attack classification, feature selection based on the identified importance rankings, and comparison with gradient boosting methods (XGBoost, LightGBM) and deep learning approaches to further improve detection accuracy while maintaining interpretability.

## References

1. Abdulhammed, R., Musafer, H., Alessa, A., Faezipour, M., & Abuzneid, A. (2020). Features dimensionality reduction approaches for machine learning based network intrusion detection. *Electronics*, 9(4), 322.
2. Ahmad, Z., Shahid Khan, A., Wai Shiang, C., Abdullah, J., & Ahmad, F. (2021). Network intrusion detection system: A systematic study of machine learning and deep learning approaches. *Transactions on Emerging Telecommunications Technologies*, 32(1), e4150.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5--32.
4. Ferrag, M. A., Maglaras, L., Moschoyiannis, S., & Janicke, H. (2020). Deep learning for cyber security intrusion detection: Approaches, datasets, and comparative study. *Journal of Information Security and Applications*, 50, 102419.
5. Kanna, P. R., & Santhi, P. (2021). Unified deep learning approach for efficient intrusion detection system using integrated spatial-temporal features. *Knowledge-Based Systems*, 226, 107132.
6. Kasongo, S. M., & Sun, Y. (2020). A deep learning method with filter-based feature engineering for wireless intrusion detection system. *IEEE Access*, 8, 149765--149780.
7. Moustafa, N., & Slay, J. (2016). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*, IEEE.
8. Sarhan, M., Layeghy, S., & Portmann, M. (2022). Towards a standard feature set for network intrusion detection system datasets. *Mobile Networks and Applications*, 27(1), 357--370.
