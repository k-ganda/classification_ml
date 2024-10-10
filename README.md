# Maternal Health Risk Prediction Model

## Introduction

The project aims to develop and compare a vanilla Neural network model to a model that uses optimization techniques to predict maternal risk as either low, mid, and high risk. 

## About the dataset

The dataset used for this model is publicly available on Kaggle:
https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data

The variables include: 

**Age:** Age in years when the woman is pregnant.

**SystolicBP:** Upper value of blood pressure (in mmHg), a significant attribute during pregnancy.

**DiastolicBP:** Lower value of blood pressure (in mmHg).

**BS:** Blood glucose levels measured in mmol/L.

**HeartRate:** Normal resting heart rate in beats per minute.

**Risk Level:** Predicted risk intensity level during pregnancy (Low, Mid, High).

## Data Cleaning and Preprocessing.

The dataset was first loaded, and no missing values were found. However, 562 duplicate rows were identified and removed to ensure accurate model performance. Upon inspecting the data, an unrealistic HeartRate value of 7 was detected and replaced with the mode, 70.

The RiskLevel variable was encoded as follows:
2 for "high risk"
1 for "mid risk"
0 for "low risk"

A correlation heatmap revealed that Blood Sugar (BS) had the strongest positive correlation with RiskLevel (0.55), while Age and HeartRate had weaker correlations. Despite this all the features were retained. There is a class imbalance (406 low-risk, 336 mid-risk, 272 high-risk). The data was split into training and testing sets and then scaled, making it ready for model training.

## Vanilla Model Implementation

This was a basic model with no optimization techniques. The model consists of two hidden layers with ReLu activation(64 and 32 neurons respectively). Softmax activation function in the output layer to handle multi-class classification.

### Results Summary

| Metric                     | Value                          |
|----------------------------|--------------------------------|
| **Test Loss**              | 0.7432                         |
| **Test Accuracy**          | 0.6703                         |
| **F1 Score**               | 0.6176                         |
| **Recall**                 | 0.6703                         |
| **Specificity per class**  | [0.55, 0.92, 0.93]            |

#### Confusion Matrix

|           | Predicted Low Risk | Predicted Mid Risk | Predicted High Risk |
|-----------|---------------------|--------------------|---------------------|
| **Actual Low Risk**   | 44                  | 2                  | 1                   |
| **Actual Mid Risk**   | 18                  | 4                  | 4                   |
| **Actual High Risk**  | 2                   | 3                  | 13                  |

#### Classification Report Summary

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **Low Risk**      | 0.69      | 0.94   | 0.79     | 47      |
| **Mid Risk**      | 0.44      | 0.15   | 0.23     | 26      |
| **High Risk**     | 0.72      | 0.72   | 0.72     | 18      |
| **Accuracy**       |           |        | 0.67     | 91      |
| **Macro Avg**      | 0.62      | 0.60   | 0.58     | 91      |
| **Weighted Avg**   | 0.62      | 0.67   | 0.62     | 91      |

## Comments

The **low** risk class exhibited a strong performance, with a high precision and recall.
The **mid** risk class performed poorly, the model struggles in correctly classifying this group.
The **high** risk classs performed reasonably well, though showing room for improvement.

Overally from the F1 score of **0.6176**, there is a balance between precision and recall but highlights the need for optimization.

## Optimization Model Implementation







