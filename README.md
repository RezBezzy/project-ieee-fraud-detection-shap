# IEEE-CIS Fraud Detection: A Financial Risk & XAI Framework

## 📌 Project Overview
This project develops a state-of-the-art fraud detection system using the **IEEE-CIS Fraud Detection** dataset. Beyond pure classification, this framework bridges the gap between **High-Performance Machine Learning** and **Operational Risk Management** (compliant with Basel III/IV frameworks).

### 🎯 Key Objectives
* **Performance:** Benchmarking Gradient Boosting (XGBoost/LightGBM) against Hybrid Deep Learning architectures.
* **Explainability (XAI):** Implementing **SHAP values** to satisfy regulatory transparency requirements (e.g., EU AI Act for high-risk financial AI).
* **Business Logic:** Translating model performance into financial impact using a **Cost-Sensitive Matrix** (Expected Loss Reduction).

---

## 🛠️ Methodology & State-of-the-Art Features

### 1. Resource-Optimized Pipeline
Although the dataset comprises ~590K transactions, the high dimensionality (450+ features) and Pandas' memory overhead can easily exceed 12GB+ RAM during complex operations. 
* **Numerical Downcasting:** Strategic reduction of bit-depth (e.g., `int64` to `int8`) based on value ranges, reducing the memory footprint by ~70% and accelerating training iterations.
* **Efficient Serialization:** Utilizing **Apache Parquet** to preserve optimized types and ensure 10x faster I/O compared to standard CSV files.

### 2. "Magic" Feature Engineering
Following industry best practices (Kaggle Top Solutions), I implemented logic-based features to bypass data anonymization:
* **User-ID (UID) Reconstruction:** Creating pseudo-UIDs using `card1`, `addr1`, and `D1n` (TransactionDay - D1) to track client behavior over time.
* **Aggregation Features:** Calculating user-specific transaction means, frequencies, and deviation scores (Z-scores) to detect anomalous spending patterns.

### 3. Modern EDA & Visualization
* **UMAP (Uniform Manifold Approximation and Projection):** Applied to identify "Relation Camouflage" and visualize high-dimensional fraud clusters in a 2D/3D space.

### 4. Hybrid Modeling Approach
* **Baseline:** Optimized Gradient Boosting (LightGBM/XGBoost) with `scale_pos_weight` to handle extreme class imbalance (3.5% fraud).
* **Deep Learning:** A **Hybrid CNN-LSTM architecture** designed to capture both spatial feature interactions and temporal transaction sequences.

### 5. Risk-Centric Evaluation
Shifting focus from pure Accuracy/AUC to economic reality:
* **Cost-Sensitive Matrix:** Optimizing the decision threshold based on the cost of **False Positives** (customer friction/churn) vs. **False Negatives** (direct financial loss/chargebacks).

---

## 📂 Project Structure
- `data/`: Optimized data storage (Parquet format, ignored by git).
- `notebooks/`: 
    - `01_Memory_Optimization.ipynb`: Downcasting and conversion to Parquet.
    - `02_EDA_and_UMAP.ipynb`: Visualizing fraud clusters and feature distributions.
    - `03_Feature_Engineering.ipynb`: UID construction and behavioral aggregations.
    - `04_Model_Training_and_XAI.ipynb`: Benchmarking and SHAP interpretability.
- `src/`: Modular Python scripts for the preprocessing pipeline.
- `reports/`: Visualization outputs, SHAP importance plots, and financial impact reports.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RezBezzy/ieee-fraud-detection-shap.git](https://github.com/RezBezzy/ieee-fraud-detection-shap.git)
   cd ieee-fraud-detection-shap
