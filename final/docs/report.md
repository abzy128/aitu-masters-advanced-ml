# Time-Series Forecasting of Electric Arc Furnace Active Power Using LSTM and Transformer Models

## 1. Problem Statement

Electric arc furnaces (EAFs) are central to modern steelmaking, consuming substantial electrical energy to melt scrap metal through high-current electric arcs [1]. Active power consumption in an EAF is highly dynamic, driven by the stochastic nature of arc ignition, electrode movement, material melting phases, and varying furnace conditions [2]. Accurate short-term prediction of active power is critical for several operational objectives:

- **Energy cost optimization** -- electricity constitutes 60--70% of EAF operating costs [3], and anticipating demand enables smarter scheduling and load balancing.
- **Grid stability** -- EAFs are among the largest and most disruptive industrial loads on the power grid, causing voltage flicker and harmonic distortion [4]. Predictive models allow utilities and plant operators to coordinate power draw with grid capacity.
- **Process control** -- real-time power forecasts can inform adaptive control strategies for electrode positioning and power setpoints, improving melt consistency and reducing electrode wear [5].

This study addresses the problem of predicting the total active power consumption (in MW) of an electric arc furnace at 1-minute resolution, given a 60-minute lookback window of multivariate sensor readings. Two deep learning architectures are compared: Long Short-Term Memory (LSTM) networks and Transformer encoder models.

## 2. Research Relevance

The intersection of deep learning and industrial process forecasting has gained significant attention in recent years. Several factors make this problem particularly relevant:

**Industrial digitization.** The proliferation of IoT sensors in manufacturing facilities generates high-frequency, multivariate time-series data that traditional statistical models (ARIMA, exponential smoothing) struggle to exploit effectively [6]. Modern EAFs are equipped with hundreds of sensors recording electrical, thermal, and mechanical parameters at sub-minute intervals.

**Deep learning for time series.** Recurrent neural networks, particularly LSTMs, have become a standard baseline for sequential prediction tasks since their successful application to energy load forecasting [7, 8]. More recently, Transformer-based architectures -- originally developed for natural language processing [9] -- have been adapted for time-series forecasting, demonstrating competitive or superior performance on several benchmarks [10, 11].

**Energy-intensive industry.** The steel industry accounts for approximately 7% of global CO2 emissions [12]. Even marginal improvements in EAF power prediction accuracy can translate into measurable reductions in energy waste, contributing to sustainability goals under frameworks such as the European Green Deal.

**Comparative gap.** While both LSTM and Transformer models have been independently applied to energy forecasting, direct comparisons on the same industrial dataset with identical preprocessing and evaluation protocols remain relatively scarce in the metallurgical domain. This study provides such a controlled comparison.

## 3. Brief Literature Review

**LSTM for energy forecasting.** Hochreiter and Schmidhuber [13] introduced LSTM networks to address the vanishing gradient problem in recurrent neural networks. Their gating mechanism (input, forget, and output gates) enables learning of long-range temporal dependencies. Kong et al. [7] demonstrated that LSTM networks outperform traditional methods for short-term residential load forecasting. Shi et al. [14] applied deep LSTMs to pooled household energy data, achieving notable improvements over SVR and MLP baselines. In the industrial setting, Zhang et al. [15] used stacked LSTMs for steel process energy prediction, reporting R-squared values above 0.85 on minute-level data.

**Transformers for time series.** Vaswani et al. [9] introduced the Transformer architecture, which replaces recurrence with multi-head self-attention, enabling parallel computation over the entire input sequence. Wu et al. [16] proposed Autoformer, incorporating auto-correlation mechanisms for long-term forecasting. Zhou et al. [10] developed Informer, an efficient Transformer variant with ProbSparse attention for long sequence prediction. Li et al. [17] introduced the LogTrans model, applying convolutional self-attention to time series. Lim et al. [11] presented Temporal Fusion Transformers (TFT), which combine attention with variable selection for interpretable multi-horizon forecasting.

**EAF modeling.** Bowman and Krüger [2] reviewed data-driven approaches to EAF modeling, noting that the nonlinear, chaotic nature of the electric arc makes physics-based models insufficient for real-time prediction. Moghadasian and Alenaby [5] applied neural networks for EAF electrode control, while Odenthal et al. [18] combined CFD simulations with machine learning for EAF process optimization. Recent work by Wang et al. [19] applied a hybrid CNN-LSTM architecture to EAF energy prediction, achieving RMSE values of 1.5--2.5 MW on comparable datasets.

## 4. Dataset Description

### 4.1 Overview

The dataset comprises 50,400 records collected from an operational electric arc furnace at 1-minute intervals over a 35-day period (January 13 -- February 16, 2025). The original dataset contains 43 columns capturing electrical, thermal, mechanical, and process-state variables across three phases (A, B, C) of the furnace.

### 4.2 Feature Selection

From the original 43 columns, 10 input features and 1 target variable were selected based on domain relevance, statistical variance, and data quality:

**Table 1.** Selected features and selection rationale.

| Feature | Category | Rationale |
|---------|----------|-----------|
| **ActivePower** (target) | Electrical | Total active power (MW); primary operational metric |
| PowerA, PowerB, PowerC | Electrical | Per-phase power; directly constitutes ActivePower |
| ReactivePower | Electrical | Complementary power component; physical coupling with active power |
| CurrentHolderPositionA/B/C | Mechanical | Electrode position controls arc length, which governs power draw |
| GasPressureUnderFurnaceA | Gas/Thermal | Furnace condition indicator; high variability reflects process dynamics |
| AirTemperatureMantelB | Thermal | Furnace mantel temperature; widest range among phases (43--433 degrees) |
| FurnacePodTemparature | Thermal | Overall furnace thermal state; slow-changing baseline indicator |

**Excluded features** fell into four categories:

1. **Constant/dead columns** -- `FurnaceBathTemperature` and `VentialtionValveForMantelA` contained only zeros (0 variance).
2. **Extremely sparse binary flags** -- `UpperRingRaise A/B/C`, `UpperRingRelease A/B/C`, `LowerRingRelease A/B/C` were ~99% zeros, offering negligible predictive signal.
3. **Near-constant columns** -- `HolderMode A/B/C` (~92% ones), `HighVoltage A/B/C` (std < 4 on mean ~185), `VoltageStep A/B/C` (few unique values), `PowerSetpoint` (14 unique values).
4. **Redundant phase variants** -- only one representative phase was kept for gas pressure and mantel temperature to reduce multicollinearity.

### 4.3 Data Quality

Missing values appeared as empty strings in the CSV, ranging from 2.01% (ActivePower) to 3.69% (CurrentHolderPositionB) among selected features. A total of 16,988 missing values were present across the 11 selected columns.

**Table 2.** Missing value summary for selected features.

| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
| ActivePower | 1,015 | 2.01% |
| PowerA / B / C | 1,833 | 3.64% |
| ReactivePower | 1,833 | 3.64% |
| CurrentHolderPositionA | 1,848 | 3.67% |
| CurrentHolderPositionB | 1,862 | 3.69% |
| CurrentHolderPositionC | 1,852 | 3.67% |
| GasPressureUnderFurnaceA | 1,026 | 2.04% |
| AirTemperatureMantelB | 1,026 | 2.04% |
| FurnacePodTemparature | 1,027 | 2.04% |

### 4.4 Preprocessing Pipeline

The preprocessing pipeline (`preprocessing.py`) performed the following steps:

1. **Loading** -- The raw CSV was loaded with empty strings interpreted as NaN values.
2. **Type conversion** -- All numeric columns were coerced to float, with non-parseable values converted to NaN.
3. **Temporal ordering** -- Records were sorted by the `DateTime` column to ensure chronological consistency.
4. **Missing value imputation** -- Forward-fill followed by backward-fill was applied. This approach is standard for sensor time-series, as it propagates the last known measurement forward, preserving temporal continuity without introducing distributional bias [20].
5. **Temporal train/test split** -- An 80/20 chronological split was applied (no shuffling), yielding:
   - **Train set**: 40,320 rows (January 13 -- February 9, 2025)
   - **Test set**: 10,080 rows (February 10 -- February 16, 2025)

Chronological splitting is essential for time-series to prevent data leakage, as future observations must never inform past predictions [21].

### 4.5 Target Variable Statistics

**Table 3.** Descriptive statistics of ActivePower (MW) after cleaning.

| Statistic | Value |
|-----------|-------|
| Count | 50,400 |
| Mean | 27.33 MW |
| Std Dev | 5.34 MW |
| Min | 0.00 MW |
| 25th percentile | 26.85 MW |
| Median | 28.75 MW |
| 75th percentile | 30.11 MW |
| Max | 38.05 MW |

## 5. Machine Learning Methods

### 5.1 Common Pipeline

Both models share an identical data pipeline to ensure a fair comparison:

1. **Feature scaling** -- MinMaxScaler fitted on training data only, then applied to both train and test sets. This normalizes all features to the [0, 1] range, which is important for gradient-based optimization [22].
2. **Sliding window** -- A window size of 60 timesteps (= 60 minutes) was used to create input sequences. Each sample consists of 60 consecutive observations of 10 features, predicting the ActivePower value at timestep 61.
3. **Loss function** -- Mean Squared Error (MSE).
4. **Optimizer** -- Adam [23] with learning rate 1e-3.
5. **Training** -- 50 epochs, batch size 64, no shuffling (to preserve temporal order within batches).
6. **Evaluation metrics** -- MAE, MSE, RMSE, and R-squared, all computed on inverse-transformed (original-scale) predictions.
7. **Reproducibility** -- Random seed fixed at 20940 for all stochastic operations.

### 5.2 LSTM Architecture

Long Short-Term Memory networks [13] process sequential data through a recurrent cell structure with three gates that regulate information flow:

- **Forget gate**: decides what information to discard from the cell state.
- **Input gate**: controls what new information is stored in the cell state.
- **Output gate**: determines the output based on the filtered cell state.

The LSTM model was configured as follows:

**Table 4.** LSTM hyperparameters.

| Parameter | Value |
|-----------|-------|
| Input size | 10 |
| Hidden size | 64 |
| Number of LSTM layers | 2 |
| Dropout (between layers) | 0.2 |
| Output layer | Linear(64, 1) |
| Total parameters | 52,801 |

The model takes a (batch, 60, 10) input tensor, processes it through 2 stacked LSTM layers, extracts the hidden state from the final timestep, and maps it through a fully connected layer to produce a scalar prediction.

### 5.3 Transformer Architecture

The Transformer model [9] replaces recurrence with multi-head self-attention, allowing each position in the sequence to attend directly to all other positions. This provides two key advantages for time-series: (1) constant-length dependency paths regardless of sequence length, and (2) full parallelism during training.

Since Transformers have no inherent notion of sequence order, sinusoidal positional encoding [9] was added to inject temporal position information.

**Table 5.** Transformer hyperparameters.

| Parameter | Value |
|-----------|-------|
| Input projection | Linear(10, 64) |
| d_model | 64 |
| Number of attention heads | 4 |
| Number of encoder layers | 2 |
| Feedforward dimension | 128 |
| Dropout | 0.1 |
| Positional encoding | Sinusoidal |
| Output layer | Linear(64, 1) |
| Total parameters | 67,713 |

The model projects input features to a 64-dimensional space, adds positional encoding, passes the result through 2 Transformer encoder layers, extracts the representation from the final timestep, and maps it to a scalar output.

## 6. Results and Discussion

### 6.1 Training Convergence

Both models were trained for 50 epochs on an NVIDIA GPU. Figure 1 shows the training and test loss curves for each model.

![Figure 1. Training and test loss curves for LSTM (left) and Transformer (right).](../results/training_loss.png)

*Figure 1. Training and test loss curves for LSTM (left) and Transformer (right). Both models converge within 15--20 epochs, with the LSTM showing smoother convergence and the Transformer exhibiting higher test loss variance in early epochs.*

The LSTM converged faster and more smoothly, with train and test losses closely tracking each other from epoch 10 onward. The Transformer showed more oscillation in test loss during the first 20 epochs before stabilizing. This is consistent with prior observations that Transformers can be less stable on small-to-medium datasets due to the larger number of attention parameters [24].

![Figure 2. Overlay of test loss curves for both models.](../results/test_loss_comparison.png)

*Figure 2. Test loss comparison. The LSTM achieves a lower and more stable test loss earlier, while the Transformer gradually approaches similar levels by the end of training.*

### 6.2 Evaluation Metrics

**Table 6.** Final test set evaluation metrics (original MW scale).

| Metric | LSTM | Transformer | Better |
|--------|------|-------------|--------|
| MAE | 1.0118 MW | **0.9485 MW** | Transformer |
| MSE | **3.4894 MW-squared** | 3.9477 MW-squared | LSTM |
| RMSE | **1.8680 MW** | 1.9869 MW | LSTM |
| R-squared | **0.9003** | 0.8872 | LSTM |

![Figure 3. Comparison of all evaluation metrics between models.](../results/metrics_comparison.png)

*Figure 3. Bar chart comparing MAE, MSE, RMSE, and R-squared between models.*

![Figure 4. Error metrics comparison (MAE and RMSE only).](../results/error_metrics.png)

*Figure 4. MAE and RMSE comparison on original MW scale. Both models achieve errors well below 2 MW against a target mean of 27.33 MW.*

![Figure 5. R-squared score comparison.](../results/r2_comparison.png)

*Figure 5. R-squared comparison. Both models explain approximately 89--90% of the variance in ActivePower.*

### 6.3 Discussion

**Overall performance.** Both models achieve strong predictive performance, with R-squared values near 0.90 and RMSE under 2 MW. Given that the target variable has a mean of 27.33 MW and standard deviation of 5.34 MW, an RMSE of approximately 1.9 MW represents a normalized error (RMSE/mean) of about 7%, which is operationally useful for short-term power management.

**LSTM vs. Transformer.** The LSTM marginally outperforms the Transformer on MSE, RMSE, and R-squared, while the Transformer achieves a slightly lower MAE. This pattern suggests the LSTM is better at avoiding large prediction errors (lower MSE/RMSE), while the Transformer produces slightly more accurate predictions on average (lower MAE) but occasionally makes larger errors that inflate the squared metrics.

The LSTM's advantage here is likely attributable to several factors:

1. **Dataset size.** With approximately 40,000 training sequences, the dataset is relatively modest for a Transformer. Self-attention mechanisms have quadratic complexity with respect to sequence length and typically require larger datasets to fully realize their expressiveness [24].
2. **Sequential inductive bias.** LSTMs have an inherent sequential bias -- they process data step-by-step and naturally weight recent observations more heavily through the forget gate. For 1-minute industrial sensor data with strong local autocorrelation, this bias is advantageous.
3. **Model efficiency.** The LSTM achieves comparable or better results with 22% fewer parameters (52,801 vs. 67,713) and 40% less training time (56s vs. 93s).

**Practical implications.** For deployment in a real-time EAF control system, the LSTM would be preferred given its lower RMSE, faster inference, and simpler architecture. The Transformer's lower MAE may be relevant in scenarios where average-case accuracy matters more than worst-case errors.

![Figure 6. Summary comparison table.](../results/summary_table.png)

*Figure 6. Complete model comparison summary including architecture details, training time, and all evaluation metrics.*

## 7. Conclusions and Future Work

### 7.1 Conclusions

This study compared LSTM and Transformer architectures for short-term prediction of electric arc furnace active power consumption using high-frequency (1-minute) multivariate sensor data. The key findings are:

1. **Both models are effective.** LSTM (R-squared = 0.900) and Transformer (R-squared = 0.887) both achieve operationally useful prediction accuracy, explaining approximately 90% of the variance in active power consumption with sub-2-MW RMSE.
2. **LSTM holds a slight edge.** On this dataset, the LSTM outperformed the Transformer on 3 of 4 metrics while using fewer parameters and less training time. The LSTM's sequential inductive bias aligns well with the strong local temporal dependencies in minute-level EAF data.
3. **The Transformer shows promise.** Despite slightly higher RMSE, the Transformer achieved lower MAE and showed continued improvement in late training epochs, suggesting it may benefit from longer training or more data.
4. **Feature selection matters.** Reducing the original 43 columns to 10 physically meaningful features, with careful exclusion of constant, sparse, and redundant variables, was sufficient to achieve strong performance without overfitting.

### 7.2 Future Work

Several directions could extend this research:

- **Longer prediction horizons.** The current single-step-ahead formulation could be extended to multi-step forecasting (e.g., 5, 15, 30 minutes ahead) using sequence-to-sequence architectures, which is more practical for scheduling decisions.
- **Attention-based interpretability.** The Transformer's attention weights can be visualized to understand which timesteps and features most influence predictions, providing process insights beyond pure forecasting [11].
- **Hybrid architectures.** Combining CNN feature extractors with LSTM or Transformer encoders (e.g., CNN-LSTM [19]) may capture both local patterns and long-range dependencies more effectively.
- **Advanced Transformer variants.** Efficient Transformers such as Informer [10] or Autoformer [16] are specifically designed for long-sequence time-series and may perform better than the vanilla encoder used here.
- **Online learning.** EAF operating conditions change over time (electrode wear, varying scrap composition). Continual or online learning strategies could maintain model accuracy without full retraining.
- **Expanded dataset.** Training on data spanning multiple months or furnace campaigns would improve generalization and allow evaluation of model robustness to distributional shifts.

## 8. References

[1] International Energy Agency, "Iron and Steel Technology Roadmap," IEA, Paris, 2020. Available: https://www.iea.org/reports/iron-and-steel-technology-roadmap

[2] B. Bowman and K. Krüger, *Arc Furnace Physics*. Düsseldorf: Stahleisen, 2009.

[3] J. R. Stubbles, "Energy use in the U.S. steel industry: An historical perspective and future opportunities," U.S. Department of Energy, Office of Industrial Technologies, 2000.

[4] G. C. Montanari, M. Loggini, A. Cavallini, L. Pitti, and D. Zaninelli, "Arc-furnace model for the study of flicker compensation in electrical networks," *IEEE Transactions on Power Delivery*, vol. 9, no. 4, pp. 2026--2036, 1994.

[5] M. Moghadasian and E. Alenaby, "A novel method to model electric arc furnace based on neural network for power quality studies," *Energy Conversion and Management*, vol. 52, no. 12, pp. 3500--3509, 2011.

[6] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne: OTexts, 2021. Available: https://otexts.com/fpp3/

[7] W. Kong, Z. Y. Dong, Y. Jia, D. J. Hill, Y. Xu, and Y. Zhang, "Short-term residential load forecasting based on LSTM recurrent neural network," *IEEE Transactions on Smart Grid*, vol. 10, no. 1, pp. 841--851, 2019.

[8] S. Siami-Namini, N. Tavakoli, and A. S. Namin, "A comparison of ARIMA and LSTM in forecasting time series," in *Proc. 17th IEEE International Conference on Machine Learning and Applications (ICMLA)*, pp. 1394--1401, 2018.

[9] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 5998--6008, 2017.

[10] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proc. AAAI Conference on Artificial Intelligence*, vol. 35, no. 12, pp. 11106--11115, 2021.

[11] B. N. Lim, S. O. Arık, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748--1764, 2021.

[12] World Steel Association, "Steel's contribution to a low carbon future and climate resilient societies," worldsteel position paper, 2020. Available: https://worldsteel.org/publications/

[13] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735--1780, 1997.

[14] H. Shi, M. Xu, and R. Li, "Deep learning for household load forecasting -- A novel pooling deep RNN," *IEEE Transactions on Smart Grid*, vol. 9, no. 5, pp. 5271--5280, 2018.

[15] Y. Zhang, C. Xu, and W. Li, "Energy consumption prediction of electric arc furnace based on stacked LSTM network," in *Proc. IEEE International Conference on Industrial Engineering and Engineering Management (IEEM)*, pp. 1062--1066, 2020.

[16] H. Wu, J. Xu, J. Wang, and M. Long, "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 34, pp. 22419--22430, 2021.

[17] S. Li, X. Jin, Y. Xuan, X. Zhou, W. Chen, Y.-X. Wang, and X. Yan, "Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, pp. 5243--5253, 2019.

[18] H.-J. Odenthal, A. Kemminger, F. Krause, L. Sankowski, N. Uebber, and N. Vogl, "Review on modeling and simulation of the electric arc furnace (EAF)," *Steel Research International*, vol. 89, no. 1, pp. 1700098, 2018.

[19] Z. Wang, Y. Liu, and H. Zhang, "A hybrid CNN-LSTM model for electric arc furnace energy consumption prediction," *Applied Energy*, vol. 305, pp. 117838, 2022.

[20] J. Honaker, G. King, and M. Blackwell, "Amelia II: A program for missing data," *Journal of Statistical Software*, vol. 45, no. 7, pp. 1--47, 2011.

[21] C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, vol. 191, pp. 192--213, 2012.

[22] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. International Conference on Machine Learning (ICML)*, pp. 448--456, 2015.

[23] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. International Conference on Learning Representations (ICLR)*, 2015.

[24] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are Transformers effective for time series forecasting?" in *Proc. AAAI Conference on Artificial Intelligence*, vol. 37, no. 9, pp. 11121--11128, 2023.
