# Anomaly detection in credit card transactions

## Overview

This project aims to detect fraudulent credit card transactions using two machine learning approaches:

- **Logistic Regression**
- **Random Forest Classifier**

The dataset used is highly imbalanced, with a significantly lower number of fraudulent transactions compared to non-fraudulent ones. To address this issue, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to balance the dataset.

---

## Dataset

The dataset consists of multiple anonymized features, including:

- `Time` - Seconds elapsed between the transaction and the first transaction in the dataset.
- `Amount` - Transaction amount.
- `Class` - Target variable (0 for non-fraudulent transactions, 1 for fraudulent transactions).

---

## Approaches Used

### 1. Logistic Regression Approach

**Steps Taken:**

1. Preprocessed the data by handling missing values and standardizing numerical features.
2. Applied **SMOTE** to handle class imbalance.
3. Trained a Logistic Regression model.
4. Evaluated performance using **accuracy, precision, recall, F1-score, and confusion matrix**.

**Challenges Faced:**

- **Class Imbalance**: Since fraudulent transactions were rare, the model tended to predict all transactions as non-fraudulent. This was handled using **SMOTE**.
- **Overfitting**: The model initially had poor generalization due to imbalanced data, which was resolved by balancing the dataset.

**Final Performance:**

- Accuracy: **Acceptable level achieved**
- Precision, Recall, and F1-score values were analyzed to check model reliability.

---

### 2. Random Forest Approach

**Steps Taken:**

1. Handled missing values and dropped highly correlated features.
2. Applied **SMOTE** to balance the dataset.
3. Trained a **Random Forest Classifier** with optimized hyperparameters:
   - `n_estimators=50` (Reduced number of trees for faster training)
   - `max_depth=8` (Controlled depth to prevent overfitting)
   - `max_features='sqrt'` (Selected features randomly for each split to improve generalization)
   - `min_samples_split=10` & `min_samples_leaf=4` (Avoided deep splits to prevent overfitting)
4. Evaluated performance using **accuracy, confusion matrix, and classification report**.

**Challenges Faced:**

- **Class Imbalance**: Like Logistic Regression, Random Forest also struggled with imbalanced classes, which was resolved using **SMOTE**.
- **Overfitting**: The initial model was overfitting due to too many trees and high depth. This was controlled by **reducing tree depth** and **limiting the number of trees**.
- **Training Time**: Random Forest took a long time to train. Reducing `n_estimators` helped speed up training.

**Final Performance:**

- Accuracy: **Balanced between training and testing to prevent overfitting**
- Classification report confirmed **good precision, recall, and F1-score** for fraud detection.

---

## Visualization

Several plots were used to analyze data and model performance:

- **Class Imbalance Visualization** (Pie Chart & Bar Graph)
- **Correlation Heatmap** to identify redundant features
- **Pair Plots & Scatter Plots** to explore data distributions
- **Confusion Matrix** to analyze model predictions
- **Residual Plots** to check errors in predictions
- **Fraud vs Non-Fraud Predictions (Pie Chart)**

---

## Conclusion

Both models were optimized to handle class imbalance and overfitting.

- **Logistic Regression** provided a simple yet interpretable model for fraud detection.
- **Random Forest** offered better accuracy with feature importance insights but required hyperparameter tuning to prevent overfitting.

**Final Recommendation:**

- If interpretability and speed are key, use **Logistic Regression**.
- If higher accuracy is needed and computational resources are available, **Random Forest** is a better choice.

---

## Running the Code

### Install Dependencies

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
```

### Execute the Script

```bash
python fraud_detection.py
```

---

## Future Improvements

- Implement **XGBoost** or **LightGBM** for better accuracy.
- Use **Anomaly Detection techniques** to detect fraud without relying on class labels.
- Deploy the model using **Flask or FastAPI** for real-time fraud detection.

---

### Author

Developed by **Teja Sri Koduru 🚀**

