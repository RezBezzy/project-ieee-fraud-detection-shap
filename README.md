# IEEE-CIS Fraud Detection & ML Explainability

## 📌 Project Overview
This project focuses on detecting credit card fraud using the **IEEE-CIS Fraud Detection** dataset. The goal is to compare various machine learning models, address extreme class imbalance, and make the predictions interpretable using **SHAP values**.

### 🎯 Background & Motivation
With my background in **Controlling and Finance**, I am particularly interested in the intersection of risk management and AI. Model explainability (XAI) is not only a technical challenge but is becoming a regulatory necessity under the **EU AI Act** (specifically regarding transparency requirements for high-risk AI systems in the financial sector).

## 🛠️ Methodology & Experiments
1.  **Explorative Data Analysis (EDA):** Identifying patterns and anomalies in fraudulent vs. legitimate transactions.
2.  **Data Preprocessing:** Handling high-cardinality categorical data and missing values.
3.  **Dimensionality Reduction (PCA):** Experimenting with **Principal Component Analysis** to reduce the feature space, handle multi-collinearity, and visualize high-dimensional clusters.
4.  **Class Imbalance:** Implementing techniques such as SMOTE (Oversampling), Undersampling, or Cost-sensitive Learning.
5.  **Modeling:** Benchmarking multiple classifiers (e.g., Logistic Regression, Random Forest, XGBoost, and LightGBM).
6.  **Interpretability (XAI):** Utilizing **SHAP (SHapley Additive exPlanations)** to analyze feature importance at both a global and local (individual transaction) level.

## 📂 Project Structure
- `data/`: Local datasets (ignored by Git due to file size).
- `notebooks/`: Jupyter Notebooks for EDA, PCA experiments, and modeling.
- `src/`: Modular Python scripts for data cleaning and engineering.
- `models/`: Serialized model checkpoints (.joblib / .pkl).
- `reports/`: Visualization outputs, SHAP plots, and project documentation.

## 🚀 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RezBezzy/ieee-fraud-detection-shap.git](https://github.com/RezBezzy/ieee-fraud-detection-shap.git)
