# 📉 Customer Churn Predictor

A full end-to-end machine learning web app that predicts whether a customer is likely to leave a service — built with Python and deployed on Streamlit.

🔗 **Live Demo:** [Click here](https://telco-churn-predictor-9v4zpcmymvfat53zs6brqe.streamlit.app/) <!-- replace with your actual link -->

---

## 🖼️ Preview

> <img width="2534" height="1280" alt="image" src="https://github.com/user-attachments/assets/97172119-57e8-40d7-8991-3bfc08159442" />


---

## 🛠️ Built with

- **Python**
- **Scikit-learn** — Logistic Regression, Decision Tree, Random Forest
- **LightGBM** — High accuracy gradient boosting models
- **Imbalanced-learn (SMOTE)** — Handle class imbalance
- **GridSearchCV** — Hyperparameter tuning
- **Matplotlib / Seaborn** — EDA and feature importance plots
- **Streamlit** — Web app interface
- **Pickle** — Model serialization

---

## 🚀 How It Works

1. User fills in customer details (demographics, services, billing)
2. Input is preprocessed — encoded and scaled to match training data
3. Tuned **LightGBM** predicts churn probability
4. App displays churn risk score + **annual revenue at risk**
5. Feature importance tab shows the top 10 drivers of churn

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Dataset | Telco Customer Churn (Kaggle, 7,043 rows) |
| Models Tried | Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM |
| Best Model | LightGBM (tuned with GridSearchCV) |
| Class Imbalance | Handled using SMOTE oversampling |
| Evaluation Metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 72.14% | 0.4844 | 0.7487 | 0.5882 | 0.8187 |
| Decision Tree | 70.58% | 0.4583 | 0.5882 | 0.5152 | 0.6683 |
| Random Forest | 76.19% | 0.5444 | 0.6390 | 0.5879 | 0.8066 |
| XGBoost | 75.48% | 0.5309 | 0.6658 | 0.5907 | 0.8023 |
| LightGBM | 75.27% | 0.5269 | 0.6818 | 0.5944 | 0.8187 |

 
### Why LightGBM?
LightGBM had the highest F1-score and ROC-AUC among all baseline models, making it the strongest candidate for this imbalanced dataset. It was then tuned using GridSearchCV across 18 parameter combinations with 3-fold CV. ROC-AUC was prioritized over raw accuracy since the dataset has significantly more non-churners than churners, making accuracy alone a misleading metric.

### Why SMOTE?
The dataset had ~73% non-churn vs ~27% churn. Without handling this imbalance, models would be biased toward predicting "No Churn." SMOTE synthetically oversamples the minority class to give a balanced training set.

---

## 📊 Feature Importance

Top factors driving churn predictions:
> <img width="1000" height="600" alt="feature_importance" src="https://github.com/user-attachments/assets/90323f5d-cceb-43c1-bf20-673c1dd3d3f3" />

## 💰 Business Impact

The app calculates the **annual revenue at risk** for each customer flagged as churn-prone:

> *"Identifying churn-prone customers early allows businesses to target them with retention offers — potentially saving thousands in lost revenue."*

---

## 📁 Project Structure

```
customer-churn/
├── app.py                              # Streamlit web app
├── Churn-predictor.ipynb               # Full training pipeline
├── model.pkl                           # Trained XGBoost model
├── encoders.pkl                        # Label encoders for categorical features
├── scaler.pkl                          # StandardScaler for numerical features
├── features.pkl                        # Feature names list
├── feature_importance.png              # Top 10 feature importance chart
├── requirements.txt                    # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/Shreeya788/Telco-Churn-Predictor.git
cd Telco-Churn-Predictor

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place it in the root folder
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Train the model first
Run Churn-predictor.ipynb 

# Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit==1.43.0
scikit-learn
pandas
numpy
xgboost
lightgbm
imbalanced-learn
optuna
matplotlib
seaborn
```

