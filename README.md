# IEEE-CIS Fraud Detection

End-to-end fraud detection pipeline on the IEEE-CIS dataset (Vesta Corporation,
~590K e-commerce transactions, 3.5% fraud rate). Covers data preprocessing,
feature engineering, model selection under a cost-sensitive business objective,
inference latency benchmarking, and SHAP-based explainability for EU AI Act
compliance.

Full methodology and results are documented in `report/fraud_detection_report.pdf`.

## Results

Four candidates were trained and independently evaluated by sweeping each
model's own cost-optimized decision threshold (not a shared 0.5 cutoff) and
by measuring real single-transaction inference latency on the same CPU.

| Model | AUC-ROC | F1 | Cost Savings vs. No Model | Inference (mean / p95) |
|---|---|---|---|---|
| LightGBM | 0.920 | 0.351 | $361,039 (59.2%) | 6.7ms / 8.4ms |
| **XGBoost** | 0.916 | **0.474** | $363,107 (59.5%) | 27.8ms / 35.4ms |
| Stacking (LightGBM+XGBoost+CatBoost) | 0.920 | 0.263 | $368,313 (60.4%) | 74.1ms / 95.8ms |
| CNN-LSTM | 0.861 | 0.218 | $275,209 (45.1%) | 243.0ms / **348.6ms** |

**XGBoost was selected as the final model.** Stacking achieves marginally
higher cost savings (+0.9pp) but at roughly 3x the inference latency and
materially worse F1, which does not justify the added complexity of
explaining a three-model ensemble. CNN-LSTM's p95 latency exceeds the 300ms
real-time requirement, disqualifying it independent of its weaker ranking
and cost performance.

A higher AUC-ROC does not always indicate the better business choice:
LightGBM has a higher AUC-ROC than XGBoost (0.920 vs. 0.916), but XGBoost
wins on cost savings, F1, and remains well within the latency budget --
which is why every candidate here is evaluated on cost savings and measured
latency, not on AUC-ROC alone.

## Key findings

- **Missing values are MNAR, in the opposite direction of the initial
  hypothesis.** Identity/device fields are missing *more often* in
  legitimate transactions, not fraudulent ones. Encoded as binary
  `_isnan` indicators regardless of direction.
- **Feature engineering, not model choice, drove most of the performance.**
  Four of the top eight features by importance are engineered (UID
  aggregations, frequency encoding, card issue date), consistent with the
  pattern reported by the original Kaggle competition winners.
- **SHAP's global ranking diverges from XGBoost's native (gain-based)
  importance** (1/10 Top-10 overlap). Investigated rather than accepted at
  face value: SHAP agrees far better with the `weight` metric (5/10 overlap,
  improving to 70% at Top 30), consistent with SHAP rewarding features used
  broadly across many trees rather than those with large gain per split.

## Repository structure

```
notebooks/
  01_Data_Cleaning_and_Downcasting.ipynb       memory optimization, Parquet conversion
  02_EDA_and_Imbalance_Analysis.ipynb           MNAR analysis, PCA on V-features, UMAP, baseline loss
  03_PCA_and_Feature_Engineering.ipynb          UID reconstruction, aggregation/frequency/time features
  04_Model_Training_and_Benchmarking.ipynb      baselines, LightGBM, XGBoost, cost-threshold optimization
  04b_Stacking_Ensemble_Colab.ipynb             stacking ensemble (run on Google Colab)
  04c_CNN_LSTM_Colab.ipynb                      CNN-LSTM benchmark (run on Google Colab, GPU)
  04d_Inference_Benchmark.ipynb                 fair CPU latency benchmark, all 4 candidates
  05_Explainability_SHAP_EU_AI_Act.ipynb        SHAP analysis and robustness checks

models/      saved model artifacts (.pkl, .keras) -- not tracked in git, see .gitignore
data/        raw and processed datasets -- not tracked in git
reports/     generated plots and CSVs referenced in the report
report/      LaTeX source and compiled PDF
```

## Setup

```bash
git clone https://github.com/RezBezzy/ieee-fraud-detection-shap.git
cd ieee-fraud-detection-shap
pip install -r requirements.txt
```

Download the IEEE-CIS dataset from
[Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) and
place `train_transaction.csv` and `train_identity.csv` in `data/`.

Run notebooks 01-04 locally, in order. Notebooks 04b and 04c are designed
for Google Colab (04c requires a GPU runtime); 04d and 05 require the
models saved by 04, 04b, and 04c to be present locally in `models/`.

## Methodology notes

- **Evaluation split**: chronological 80/20 (no shuffle), since random
  shuffling would leak future transactions into training. The official
  Kaggle test set could not be used for additional validation -- its
  labels were never publicly released.
- **Class imbalance**: handled via `scale_pos_weight`, not SMOTE, to avoid
  generating synthetic transactions across 421 correlated/categorical
  features.
- **Model selection criterion**: cost savings against a no-model baseline
  (FN = transaction amount, FP = $10), evaluated independently per
  candidate at its own optimal threshold, per the project's decision
  framework -- not a single shared threshold or AUC-ROC ranking.

## Limitations

CNN-LSTM treats each transaction as an independent feature vector rather
than a true per-user sequence (average history per UID is 2.2
transactions, too short for sequence modelling to be fairly tested). The
SHAP/native-importance divergence is substantially, but not completely,
explained by the gain-vs-weight distinction. See the report's Limitations
section for the full list.
