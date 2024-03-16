# Online Food Sales Analysis

## Overview

This project aims to analyze customer retention in the online food sales industry. By employing various machine learning models and data preprocessing techniques, this project seeks to understand the factors that contribute to customer retention and churn. The dataset used in this project is sourced from Kaggle and contains various features related to customer transactions and interactions.

## Dataset

The dataset used in this project is [Data Source](https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset/data).

It contains the following features: 

1. Age
2. Gender
3. Marital Status
4. Occupation
5. Monthly Income
6. Educational Qualifications
7. Family size
8. Latitude
9. Longitude
10. Pin code
11. Output
12. Feedback
13. Unnamed: 12

The main variables of interest is **Output**.

## Requirements

The following libraries are required to run the notebook:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Key Features

1. Exploratory Data Analysis (EDA): The project features a stage of Exploratory Data Analysis (EDA), where we examine the data closely to identify customer reordering trends 
2. Classification: The project employs a variety of models, including Logistic Regression, Random Forest,and K-Nearest Neighbors to predict if the customer is going to order again

## Results

1. **Logistic Regression**:

   a. Precision for predicting "Yes" (0.90) is higher than for predicting "No" (0.63),indicating that the model is better at correctly identifying positive cases.
   
   b. Recall for predicting "Yes" (0.93) is higher than for predicting "No" (0.55),indicating that the model is better at capturing actual positive cases.
   
   c. F1-score for predicting "Yes" (0.91) is high,indicating a good balance between precision and recall for positive cases.
   
   d. ROC AUC score (0.857) is also quite good,indicating that the model performs well in distinguishing between positive and negative cases.

3. **Random Forest**:
   
   a. Precision,recall,and F1-score for predicting "Yes" are all high (0.91,0.96,0.93 respectively),indicating that the model performs well in identifying positive cases.
   
   b. Precision,recall,and F1-score for predicting "No" are lower compared to Logistic Regression,indicating that the model is not as good at predicting negative cases.
   
   c. ROC AUC score (0.913) is higher than Logistic Regression,indicating better overall performance in distinguishing between positive and negative cases.

5. **K-Nearest Neighbors**:
   
   a. Precision,recall,and F1-score for predicting "Yes" are high (0.92,0.95,0.93 respectively),similar to Random Forest.
   
   b. Precision,recall,and F1-score for predicting "No" are lower compared to Logistic Regression and Random Forest.
   
   c. ROC AUC score (0.839) is lower than both Logistic Regression and Random Forest,indicating that the model is not as effective in distinguishing between positive and negative cases.

### Overall:

1. Random Forest has the highest F1-score and ROC AUC score,indicating better overall performance among the three models.
2. Logistic Regression performs reasonably well but is outperformed by Random Forest in most metrics.
3. K-Nearest Neighbors lags behind the other two models,particularly in distinguishing between positive and negative cases (ROC AUC score).

