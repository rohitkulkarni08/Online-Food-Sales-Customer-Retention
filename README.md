# Online Food Sales Analysis

## Overview

This project focuses on Exploratory Data Analysis (EDA) and modeling to uncover insights from the data and build predictive models to understand Customer Churn.

Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company or service. It is a critical metric for businesses as it directly impacts revenue and profitability. 
High churn rates can indicate dissatisfaction with the product or service, poor customer experience.

## Dataset

The dataset used in this project is [Data Source](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction/data).

It contains the following features: 

 1. RowNumber
 2. CustomerId
 3. Surname
 4. CreditScore
 5. Geography
 6. Gender
 7. Age
 8. Tenure
 9. Balance
 10. NumOfProducts
 11. HasCrCard
 12. IsActiveMember
 13. EstimatedSalary
 14. Exited

The main variables of interest is **Exited**.

## Requirements

The following libraries are required to run the notebook:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Key Features

1. Handling Imbalanced Data: The project implements techniques to help handle imbalanced data such as SMOTE, ensuring accurate predictions even when the dependent variable is underrepresented.
2. Exploratory Data Analysis (EDA): The project features a stage of Exploratory Data Analysis (EDA), where we examine the data closely to identify trends and understand the reasons behind customer churn.
3. Classification: The project employs a variety of models, including Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine, XGBoost, and Gradient Boosting, to predict customer churn, with techniques such as class weighting and SMOTE used to handle class imbalance.

## Results

| Model                   | Accuracy | Recall Score | F1 Score | ROC AUC Score |
|-------------------------|----------|--------------|----------|---------------|
| Logistic Regression     | 0.703667 | 0.683219     | 0.473029 | 0.764076      |
| Random Forest           | 0.862000 | 0.414384     | 0.538976 | 0.852447      |
| K-Nearest Neighbors     | 0.752333 | 0.667808     | 0.512147 | 0.776639      |
| Support Vector Machine  | 0.785667 | 0.662671     | 0.546224 | 0.822503      |
| XGBoost                 | 0.833000 | 0.609589     | 0.586974 | 0.841784      |
| Gradient Boosting       | 0.817000 | 0.700342     | 0.598391 | 0.859767      |

From the results of the classification models on the churn prediction dataset, we can infer the following:

1. **Gradient Boosting** has the highest F1 score (0.598391) and the highest ROC AUC score (0.859767) among all the models. This suggests that Gradient Boosting is the most effective model in balancing precision and recall and has the best ability to distinguish between the churned and non-churned customers.

2. **XGBoost** also performs well, with a relatively high F1 score (0.586974) and a good ROC AUC score (0.841784). This indicates that XGBoost is another strong model for this task.

3. **Random Forest** has a high accuracy (0.862000) but a lower F1 score (0.538976) compared to Gradient Boosting and XGBoost. This suggests that while Random Forest is good at predicting the majority class (non-churned customers), it might not be as effective at identifying the minority class (churned customers).

4. **Support Vector Machine** and **K-Nearest Neighbors** have moderate F1 scores and ROC AUC scores. They perform better than Logistic Regression but are not as effective as Gradient Boosting or XGBoost for this dataset.

5. **Logistic Regression** has the lowest accuracy (0.703667), F1 score (0.473029), and ROC AUC score (0.764076) among all the models. This indicates that Logistic Regression is the least effective model for predicting customer churn in this dataset.

#### Overall:

Gradient Boosting appears to be the best model for this churn prediction task, followed closely by XGBoost. These models are able to better handle the class imbalance and provide a good balance between precision and recall. 
