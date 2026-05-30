# Mobile Money Fraud Detection

A machine learning pipeline for detecting financial fraud in mobile money transactions using the PaySim synthetic dataset. The project covers the full workflow: exploratory data analysis, preprocessing, feature engineering, model training with cross-validation, and a live prediction interface.

**Live Demo:** [Fraud Detection System](https://paysim-fraud-detection.streamlit.app) 

---

## Dataset

PaySim simulates mobile money transactions based on real logs from a financial provider in Africa.

- Source: [Kaggle — ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Full size: 6,362,620 transactions
- Working subset: 200,000 rows (stratified)
- Fraud rate: 0.13%

---

## Pipeline

| Step | Description |
|------|-------------|
| Stratified sampling | 200k rows preserving fraud ratio |
| EDA | Class imbalance, fraud by type, distributions, correlations |
| Preprocessing | Drop leaky columns, median imputation, OHE, IQR capping on normal class only |
| Feature engineering | 5 domain-informed features (see below) |
| CV + SMOTE | Stratified 5-Fold, SMOTE applied inside each fold only |
| Baseline models | 5 classifiers evaluated on F1, ROC-AUC, PR-AUC |
| Feature selection | MI filter + Tree importance, both inside CV pipeline |
| Tuning | GridSearchCV on best model, threshold optimization for F1 |
| Deployment | Streamlit live prediction app |

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `amount_ratio_orig` | Ratio of transaction amount to origin balance — strongest predictor by both MI and Tree importance |
| `orig_balance_zeroed` | Binary: origin balance dropped to zero after transaction |
| `dest_balance_unchanged` | Binary: destination balance unchanged despite receiving funds |
| `balance_error_orig` | Absolute discrepancy in origin account balance |
| `balance_error_dest` | Absolute discrepancy in destination account balance |

---

## Results

Best model: **AdaBoost**

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9728 |
| ROC-AUC | 0.9999 |
| Recall | 1.0000 |
| F1 (at threshold 0.56) | 0.9135 |
| Optimal threshold | 0.56 |

Fraud occurs exclusively in `TRANSFER` and `CASH_OUT` transaction types.

---

## Project Structure

```
paysim-fraud-detection/
├── Code_.ipynb       # Full pipeline notebook
├── app.py                 # Streamlit demo
├── requirements.txt
├── fraud_model.pkl        # Trained AdaBoost pipeline
├── model_features.pkl     # Feature column order
├── best_threshold.pkl     # Optimal decision threshold
└── README.md
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The three `.pkl` files must be in the same directory as `app.py`. Generate them by running `Code_.ipynb` in Google Colab with Kaggle credentials set as Secrets.



