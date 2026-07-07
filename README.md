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
| **LightGBM** | **0.915** | 0.332 | **$327,398 (53.7%)** | **2.53ms / 3.07ms** |
| XGBoost | 0.912 | **0.432** | $322,698 (52.9%) | 35.80ms / 57.48ms |
| Stacking (LightGBM+XGBoost+CatBoost) | 0.913 | 0.256 | $331,587 (54.4%) | 84.99ms / 115.53ms |
| CNN-LSTM | 0.838 | 0.192 | $160,609 (26.3%) | 115.49ms / 213.14ms |

**LightGBM was selected as the final model.** Stacking achieves marginally
higher cost savings (+0.7pp) but at roughly 33x the inference latency and
materially worse F1, which does not justify the added complexity of
explaining a three-model ensemble. All four candidates satisfy the 300ms
real-time requirement on both mean and p95 latency.

The cost savings gap between LightGBM, XGBoost, and Stacking is narrow
(52.9%--54.4%), making latency the decisive tie-breaker: LightGBM at
2.53ms is 14x faster than XGBoost and 33x faster than Stacking, while
capturing nearly all of Stacking's cost-saving potential.

## Key findings

- **Missing values are MNAR, in the opposite direction of the initial
  hypothesis.** Identity/device fields are missing *more often* in
  legitimate transactions, not fraudulent ones. Encoded as binary
  `_isnan` indicators regardless of direction.
- **Feature engineering, not model choice, drove most of the performance.**
  Four of the top eight features by importance are engineered (UID
  aggregations, frequency encoding, card issue date), consistent with the
  pattern reported by the original Kaggle competition winners.
- **SHAP's global ranking diverges from LightGBM's native importance**
  (4/10 Top-10 overlap with gain-based importance). Investigated rather
  than accepted at face value: the direction of best agreement is
  model-dependent -- for LightGBM, gain agrees better than weight/split
  (6/10 vs 4/10 overlap), the opposite of the pattern typically seen for
  XGBoost. Agreement improves to 83% at Top 30, confirming a robust
  pattern rather than coincidence.

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
  generating synthetic transactions across 340 correlated/categorical
  features.
- **Model selection criterion**: cost savings against a no-model baseline
  (FN = transaction amount, FP = $10), evaluated independently per
  candidate at its own optimal threshold, per the project's decision
  framework -- not a single shared threshold or AUC-ROC ranking.

## Limitations

CNN-LSTM treats each transaction as an independent feature vector rather
than a true per-user sequence (average history per UID is 2.1
transactions, too short for sequence modelling to be fairly tested). The
feature engineering pipeline was corrected to compute all aggregation
statistics (UID features, frequency encodings, PCA) exclusively on the
training partition -- an earlier version computed these on the full dataset,
introducing a leakage risk that has since been addressed. See the report's
Limitations section for the full list.
