# Maternal Health Risk Prediction Model

## Introduction

The project aims to develop and compare a vanilla Neural network model to a model that uses optimization techniques to predict maternal risk as either low, mid, or high risk. 

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

A correlation heatmap revealed that Blood Sugar (BS) had the strongest positive correlation with RiskLevel (0.55), while Age and HeartRate had weaker correlations. Despite this, all the features were retained. There is a class imbalance (406 low-risk, 336 mid-risk, 272 high-risk). The data was split into training and testing sets and then scaled, making it ready for model training.

## Vanilla Model Implementation

This was a basic model with no optimization techniques. It consists of two hidden layers with ReLu activation(64 and 32 neurons, respectively). The output layer has a softmax activation function to handle multi-class classification.

### Results Summary

| Metric                     | Value                          |
|----------------------------|--------------------------------|
| **Test Loss**              | 0.7978                         |
| **Test Accuracy**          | 0.6374                         |
| **F1 Score**               | 0.6001                         |
| **Recall**                 | 0.6374                         |
| **Specificity per class**  | [0.50, 0.87, 0.95]             |

#### Confusion Matrix

|           | Predicted Low Risk | Predicted Mid Risk | Predicted High Risk |
|-----------|---------------------|--------------------|---------------------|
| **Actual Low Risk**   | 42                  | 4                  | 1                   |
| **Actual Mid Risk**   | 19                  | 5                  | 2                   |
| **Actual High Risk**  | 3                   | 4                  | 11                  |

#### Classification Report Summary

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **Low Risk**      | 0.66      | 0.89   | 0.76     | 47      |
| **Mid Risk**      | 0.38      | 0.19   | 0.26     | 26      |
| **High Risk**     | 0.79      | 0.61   | 0.69     | 18      |
| **Accuracy**       |           |        | 0.64     | 91      |
| **Macro Avg**      | 0.61      | 0.57   | 0.57     | 91      |
| **Weighted Avg**   | 0.60      | 0.64   | 0.60     | 91      |

## Comments

The **low** risk class exhibited a good performance.
The **mid** risk class performed poorly, the model struggles in correctly classifying this group.
The **high** risk class performed reasonably well, though showing room for improvement.

Overall, the F1 score of **0.6001** shows a balance between precision and recall but highlights the need for optimization.

## Optimization Model Implementation

To enhance the vanilla model’s performance, various optimization techniques were applied while keeping the original layer structure intact. In this project, the optimization techniques explored were: L1 Regularisation, L2 Regularisation, Early stopping and two optimizers(Adam and RMSprop). Each of these played a vital role in ensuring robust learning from our dataset.

The first optimization approach involved applying L1 regularization in conjunction with both the Adam and RMSprop optimizers. To mitigate the risk of overfitting, early stopping was implemented. This included careful tuning of key parameters: learning rate, patience value, and the strength of the L1 regularization. The performance of each configuration was meticulously noted to identify the optimal settings.

### L1 regularisation with RMSprop

Through numerous trials, a stable configuration was identified for the L1 + RMSprop model. The optimal settings were as follows:

Patience: 10

Epochs: 100

Batch Size: 32

Learning Rate: 0.001

L1 Regularizer Strength: 0.0001 for both layers

This configuration achieved a test accuracy of **70%** and a test loss of **0.7340**.

To further evaluate model performance, a confusion matrix and classification report were generated.

The model correctly predicted 46 instances of low risk, 3 instance of mid risk, and 15 instances of high risk. This is an improvement from the vanilla model for low risk and high risk however a struggle is seen in classifying mid risk.

### L1 Regularization with Adam

For this, optimal settings were slightly different. 

Patience: 10

Epochs: 100

L1 Regularizer Strength: 0.0001

Learning Rate: 0.01

This yielded a test accuracy of **65%** and test loss of **0.7575**. The performance is slightly lower than that with RMSprop. The model correctly predicted **44** instances of low risk, **1** instance of mid risk and **14** instances of high risk. However, it misclassified 3 mid-risk instances as low risk and 21 low-risk instances as mid risk, showcasing areas for improvement.

### L2 Regularization with RMSprop

L2 regularization was implemented to reduce overfitting by penalizing larger weights during training. This model utilized a learning rate of 0.01 and l2 regularisation strength of 0.0001 across all layer. 

This yielded a test accuracy of **71%** and a test loss of **0.7327**. The model correctly predicted **46** instances of low risk, **3** instances of mid risk and **14** instances of high risk. So far, this is the best performing model compared to vanilla and L1 regularisation.

### L2 regularisation with Adam

This implemented L2 with adam optimiser. The model achieved a test accuracy of **71%** and test loss of **0.7098**. he model correctly predicted **46** instances of low risk, **4** instances of mid risk and **15** instances of high risk outperforming the one with RMSprop.

This model therefore provided a balanced performance with high accuracy of **3%** increase from the vanilla model and a better performance overally.

## NOTE

It is essential to note that all results from the various models were derived using the default threshold from the softmax activation function. To better address mid-risk classifications, adjustments to the class thresholds were considered. However, the process of selecting an appropriate threshold proved complex, as it required finding a balance that would enhance mid-risk performance without compromising the accuracy of high and low-risk classifications. Ultimately, we opted to maintain the default classification method, which assigns classes based on the highest probability value.

Below, is a summary of the results from the different implementations

| Metric                          | **L1 + Adam**                | **L1 + RMSprop**             | **L2 + Adam** 🏆            | **L2 + RMSprop**             |
|----------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| **Test Loss**                    | 0.7575                       | 0.7340                       | 0.7098                       | 0.7327                       |
| **Test Accuracy**                | 0.6484                       | 0.7033                       | 0.7143                       | 0.7143                       |
| **F1 Score**                     | 0.5734                       | 0.6325                       | 0.6509                       | 0.6389                       |
| **Recall**                       | 0.6484                       | 0.7033                       | 0.7143                       | 0.7143                       |
| **Specificity (Low Risk)**       | 0.5000                       | 0.5227                       | 0.5227                       | 0.5000                       |
| **Specificity (Mid Risk)**       | 0.9077                       | 0.9692                       | 0.9846                       | 1.0000                       |
| **Specificity (High Risk)**      | 0.9452                       | 0.9452                       | 0.9452                       | 0.9452                       |
| **Precision (Low Risk)**         | 0.67                         | 0.69                         | 0.69                         | 0.68                         |
| **Precision (Mid Risk)**         | 0.14                         | 0.60                         | 0.80                         | 1.00                         |
| **Precision (High Risk)**        | 0.78                         | 0.79                         | 0.79                         | 0.80                         |
| **Recall (Low Risk)**            | 0.94                         | 0.98                         | 0.98                         | 0.98                         |
| **Recall (Mid Risk)**            | 0.04                         | 0.12                         | 0.15                         | 0.12                         |
| **Recall (High Risk)**           | 0.78                         | 0.83                         | 0.83                         | 0.83                         |
| **Macro Average Precision**      | 0.53                         | 0.69                         | 0.77                         | 0.77                         |
| **Macro Average Recall**         | 0.58                         | 0.64                         | 0.65                         | 0.64                         |
| **Macro Average F1 Score**       | 0.54                         | 0.60                         | 0.66                         | 0.63                         |
| **Weighted Average Precision**   | 0.54                         | 0.68                         | 0.74                         | 0.74                         |
| **Weighted Average Recall**      | 0.65                         | 0.70                         | 0.71                         | 0.71                         |
| **Weighted Average F1 Score**    | 0.57                         | 0.63                         | 0.73                         | 0.70                         |



## Running and loading Saved models

Both the vanilla model and the L2 regularization + Adam model were saved for future use. To load these models and make predictions, follow these steps:

1. Open the notebook and run all the cells from the section titled "Loading the Libraries" up to the "Data Preprocessing" section. This includes splitting the data and scaling it appropriately.

2. Once the preprocessing is complete, proceed to the section labeled "LOADING THE SAVED MODEL" and run the corresponding cell. This cell contains pre-written code to load the saved models and facilitate prediction.

3. Ensure that both the model files and the dataset are uploaded in the appropriate directories for smooth execution.
   
By following these steps, you will be able to successfully load the saved models and continue with prediction tasks.

## Conclusion 

The optimization techniques applied, particularly L2 regularization combined with the Adam optimizer, significantly improved the performance of the maternal health risk prediction model compared to the vanilla model. Despite these improvements, the mid-risk class remained challenging for all models, as reflected by lower precision and recall metrics. Optimization helped mitigate overfitting and enhanced generalization, but further refinement is needed for more accurate mid-risk predictions.

## Recommendations

To improve the model's ability to classify mid-risk cases, gathering a more balanced dataset with additional and diverse samples for the mid-risk class is essential. Collaborative efforts and further insights on addressing this issue are encouraged to enhance the model's overall performance.










