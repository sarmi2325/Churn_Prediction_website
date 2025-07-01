# 📈 Telecom Churn Prediction using Linear Models

This project focuses on building interpretable and reliable linear models to predict customer churn for a telecom company. By experimenting with various preprocessing strategies, feature engineering, and class imbalance handling, we identify the best linear model before transitioning into a more advanced stacked ensemble.

---

## Problem Statement

Customer churn prediction is crucial for telecom companies aiming to reduce customer loss. Our objective is to use **linear models** to create a fast, interpretable, and fairly accurate model that predicts whether a customer is likely to churn.

---

## 🛠️ Workflow

### ✅ Step 1: Baseline Linear Models

trained four basic models using default hyperparameters and minimal preprocessing:

| Model                  | ROC-AUC  | Accuracy |
|-----------------------|----------|----------|
| Logistic Regression   | **0.835** | 73%     |
| SVM                   | 0.817    | 69%     |
| Naive Bayes           | 0.810    | 64%     |
| K-Nearest Neighbors   | 0.767    | 75%     |

➡ **Logistic Regression** showed the best balance between precision and recall.

---

### ✂️ Step 2: Feature Selection & Class Imbalance Handling

- **Top features** selected using SHAP values and correlation analysis.
- **SMOTE** used for balancing the dataset where applicable.

| Model Variant          | ROC-AUC  | Accuracy |
|------------------------|----------|----------|
| Logistic Regression (Top Features) | 0.830 | 72% |
| SVM (Top Features)     | 0.828    | 64%     |
| KNN (SMOTE)            | 0.758    | 69%     |
| Naive Bayes (SMOTE)    | 0.810    | 66%     |

---

### 🧪 Step 3: Feature Engineering

Created domain-driven features:
- `avg_monthly_spend`
- `num_services`
- `is_new_customer`
- `has_full_streaming`
- Contract indicators, etc.

| Model Variant          | ROC-AUC  | Accuracy |
|------------------------|----------|----------|
| Logistic Regression (Top Features) | **0.838** | 73% |
| SVM (Top Features)     | 0.823    | 69%     |
| KNN (SMOTE)            | 0.769    | 69%     |
| Naive Bayes (SMOTE)    | 0.816    | 69%     |

---

### 🔍 Step 4: Regularization (L1 vs L2)

Tested L1 and L2 penalties on Logistic Regression:

| Penalty      | ROC-AUC | Accuracy | Interpretation |
|--------------|---------|----------|----------------|
| **L1**       | 0.834   | 73%      | Sparse (Top 15 features) |
| **L2**       | **0.834** | 72%    | All features retained     |

✅ **Final choice: L2-regularized Logistic Regression** for performance and stability.

---

## ✅ Final Model Metrics


Confusion Matrix:
[[720 313]
 [ 74 300]]

Precision (Churn): 0.49
Recall (Churn):    0.80
F1 Score:          0.61
Accuracy:          73%
ROC-AUC Score:     0.834

# Telecom Churn Prediction using Tree-Based Models


## 🛠️ Workflow

### ✅ Step 1: Baseline Tree Models

trained three baseline tree-based classifiers on the preprocessed dataset:

| Model            | ROC-AUC  | Accuracy |
|------------------|----------|----------|
| Random Forest    | 0.842    | 75%     |
| XGBoost          | 0.847    | 76%     |
| LightGBM         | **0.850** | 76%     |

➡ **LightGBM** gave the best balance between recall and precision.

---

### ✂️ Step 2: Feature Selection & Imbalance Handling

- SHAP used to identify top 15 important features.
- SMOTE + Tomek applied to handle class imbalance.

| Model (Top Features) | ROC-AUC  | Accuracy |
|----------------------|----------|----------|
| LightGBM (Top 15)    | 0.821    | 74%     |
| XGBoost (Top 15)     | 0.818    | 74%     |
| Random Forest (Top 15)| 0.815   | 74%     |

---

### 🧪 Step 3: Feature Engineering

incorporated advanced domain-specific features to improve signal:

- `avg_monthly_spend`
- `is_new_customer`
- `has_full_streaming`
- `Contract_One year`, `Contract_Two year`
- `InternetService_Fiber optic`
- etc.

📌 These features consistently appeared in SHAP value top contributors.

---

## ✅ Final Model Metrics


Best Performing: LightGBM (Top Features + Feature Engineering + SMOTE)
Confusion Matrix:
[[767 266]
 [ 95 279]]

Precision (Churn): 0.51
Recall (Churn):    0.75
F1 Score:          0.61
Accuracy:          74%
ROC-AUC Score:     0.820

## 🧪 Stacked Model Architecture

used **StackingClassifier** with:

- **Base Models**:  
  - Logistic Regression (with L2 penalty)  
  - LightGBM (gradient boosting decision tree)

- **Meta Model**:  
  - Logistic Regression (on top of base predictions)

---

## 📊 Evaluation Metrics


Confusion Matrix:
[[767 266]
 [ 95 279]]

Precision (Churn):     0.51  
Recall (Churn):        0.75  
F1 Score:              0.61  
Accuracy:              74%  
ROC-AUC Score:         0.820
