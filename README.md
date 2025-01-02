## Project Overview

The goal of this project is to build a machine learning algorithm to predict and detect obesity using various demographic, behavioral, and dietary variables. The project compares the performance of two popular machine learning models, **Random Forest** and **XGBoost**, to determine the most effective approach for this task.

## Problem Statement

Obesity is a global health issue that requires effective strategies for early detection and management. By leveraging machine learning techniques, we aim to develop a predictive model that can identify individuals at risk of obesity based on several input features. This will help in proactive health interventions and decision-making.

---

## Dataset Description

The dataset used for this project contains information on various factors related to obesity, including demographic details, family history, eating habits, and physical activity. Below are some key details:

- **File Path**: `C:\Users\user\Desktop\Projects\Obesity Predictor App\obesity new.csv`
- **Features**:
  - **Numerical Features**: Physical activity levels, family history, and calorie intake.
  - **Categorical Features**: Gender, eating habits, and other behavioral factors.
- **Target Variable**: `nobeyesdad` (categorical) — Represents the obesity classification of individuals.

### Data Preprocessing
1. Checked for missing values and duplicates, and handled them appropriately.
2. Resampled the data using undersampling techniques to balance the dataset.
3. Preprocessed numerical and categorical features using:
   - **Numerical Pipeline**: Median imputation and standard scaling.
   - **Categorical Pipeline**: One-hot encoding and ordinal encoding.

---


## Comparison of Random Forest and XGBoost Models

Both the **Random Forest** and **XGBoost** models perform well in the classification task, but they differ in their feature importance distributions, model performance, and strengths. Here’s a detailed comparison:

## 1. Feature Importances

### Random Forest
- The most important feature is `num_pipe__family_history` with an importance of **0.07**. However, the importance values of other features are relatively smaller.
- Numerical features like `num_pipe__caec`, `num_pipe__favc`, and `num_pipe__calc` are significant but less impactful compared to the XGBoost model.
- Categorical features, such as `cat_pipe2__ch2o_2`, `cat_pipe2__ncp_3`, and `cat_pipe2__fcvc_2`, contribute moderately, but overall, the model places less emphasis on categorical variables.

### XGBoost
- `num_pipe__family_history` has the highest importance with **0.25**, significantly outpacing Random Forest’s highest importance. This suggests XGBoost relies more heavily on this feature.
- XGBoost also assigns higher importance to other numerical features like `num_pipe__favc`, `num_pipe__caec`, and `num_pipe__public_transportation`, with importance values between **0.05 and 0.03**.
- Categorical features like `cat_pipe2__ncp_4`, `cat_pipe2__age_21`, and `cat_pipe2__ch2o_2` have comparable importance to Random Forest, indicating that both models are sensitive to categorical variables, but XGBoost appears to use them more effectively.

**Key Insight:**
- **XGBoost** gives more weight to the most important features (e.g., `num_pipe__family_history`), making it more confident in its predictions. It generally utilizes both numerical and categorical features more effectively than Random Forest.

---

## 2. Model Performance

### Random Forest
- **Accuracy**: Not explicitly mentioned but inferred from the confusion matrix and F1 score. The **F1 score** is **0.88**, indicating decent performance.
- **Confusion Matrix**:
  - True Negatives (TN): 164
  - False Positives (FP): 29
  - False Negatives (FN): 19
  - True Positives (TP): 177
- The Random Forest model has a slightly higher number of false positives compared to XGBoost, with **29 FP** versus **23 FP** for XGBoost.

### XGBoost
- **Accuracy**: **89%**, indicating that it correctly predicted the class for 89% of instances.
- **Confusion Matrix**:
  - True Negatives (TN): 170
  - False Positives (FP): 23
  - False Negatives (FN): 20
  - True Positives (TP): 176
- **Precision**: Both classes show a precision of **0.89**, meaning that XGBoost performs well when it predicts positive instances.
- **Recall**: Class 1 has a slightly higher recall (**0.90**) than class 0 (**0.88**), indicating that the model is better at identifying instances of class 1.
- **F1-Score**: The F1-score for both classes is **0.89**, showing a good balance between precision and recall.

**Key Insight:**
- **XGBoost** outperforms Random Forest with higher accuracy (**89%** vs. **88%** inferred from F1 score), fewer false positives, and a better balance between precision and recall.

---

## 3. Overall Performance and Suitability

### Random Forest
- Random Forest is a robust model, performing well in general classification tasks. However, its reliance on a larger number of less important features could lead to a less interpretable model.
- It shows a slight imbalance with more false positives, but overall, its F1 score of **0.88** suggests it can be a good choice for tasks where interpretability is key.

### XGBoost
- XGBoost provides a stronger overall performance, with a higher accuracy of **89%** and a better balance of precision and recall. It is better at handling feature importance and emphasizes the most significant predictors.
- XGBoost is particularly suited for tasks where high accuracy is essential, and the model can tolerate slight complexity in interpretation, especially for making critical decisions.

---

## Conclusion

- **XGBoost** is the better model in this case due to its **higher accuracy** (**89%** vs. **88%**), **better precision/recall balance**, and **stronger feature importance management**.
- **Random Forest**, while also a solid model, would be more useful in cases where **model interpretability** is a top priority or if there are concerns about training time, as Random Forest typically requires more time to train on large datasets compared to XGBoost.

If performance and predictive accuracy are the primary concerns, **XGBoost** is the better choice.

