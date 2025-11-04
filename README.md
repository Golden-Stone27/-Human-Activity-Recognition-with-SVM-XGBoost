# 🧠 Human Activity Recognition using SVM & XGBoost

This project aims to classify **human activities** using **accelerometer-based time series data**.  
The data is preprocessed, cleaned, and transformed into statistical features before being used to train and compare two models: **SVM** and **XGBoost**.

---

## 📑 Table of Contents
- [1. Library Imports and Data Loading](#1-library-imports-and-data-loading)
- [2. Dataset Overview and Initial Inspection](#2-dataset-overview-and-initial-inspection)
- [3. Missing Value Analysis and Basic Visualizations](#3-missing-value-analysis-and-basic-visualizations)
- [4. Outlier Detection using IQR Method](#4-outlier-detection-using-iqr-method)
- [5. Feature Extraction using Time Windows](#5-feature-extraction-using-time-windows)
- [6. Feature Distribution Visualization](#6-feature-distribution-visualization)
- [7. Data Standardization and Train-Test Split](#7-data-standardization-and-train-test-split)
- [8. SVM Model Training and Evaluation](#8-svm-model-training-and-evaluation)
- [9. SVM Confusion Matrix Visualization](#9-svm-confusion-matrix-visualization)
- [10. Hyperparameter Optimization for SVM (GridSearchCV)](#10-hyperparameter-optimization-for-svm-gridsearchcv)
- [11. Label Encoding for Activity Classes](#11-label-encoding-for-activity-classes)
- [12. XGBoost Model Training and Evaluation](#12-xgboost-model-training-and-evaluation)
- [13. Hyperparameter Optimization for XGBoost](#13-hyperparameter-optimization-for-xgboost)
- [14. Model Performance Comparison (Accuracy & F1 Score)](#14-model-performance-comparison-accuracy--f1-score)

---

'''python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("time_series_data_human_activities.csv", header=None)
'''

2. Dataset Overview and Initial Inspection

The structure of the dataset is explored. Column names are added, and a few sample rows are displayed.

3. Missing Value Analysis and Basic Visualizations

Missing values are checked, and the class distributions across users and activities are visualized.

4. Outlier Detection using IQR Method

Outliers in accelerometer data are removed using the Interquartile Range (IQR) method.

'''python
Q1 = df[['x','y','z']].quantile(0.25)
Q3 = df[['x','y','z']].quantile(0.75)
IQR = Q3 - Q1
df_clean_iqr = df[~((df[['x','y','z']] < (Q1 - 1.5 * IQR)) | (df[['x','y','z']] > (Q3 + 1.5 * IQR))).any(axis=1)]
'''

5. Feature Extraction using Time Windows

The raw time series is divided into sliding windows, and for each window, the following features are extracted:

Mean

Standard deviation

Maximum and minimum

Energy (sum of squares of x, y, z)

6. Feature Distribution Visualization

Feature distributions are visualized using seaborn.pairplot to show how activities differ in feature space.

7. Data Standardization and Train-Test Split

All features are standardized using StandardScaler.
Data is then split into training and testing sets (80% / 20%).

8. SVM Model Training and Evaluation

An SVM classifier with an RBF kernel is trained on the standardized data.
Accuracy and classification metrics are reported.

9. SVM Confusion Matrix Visualization

The confusion matrix is visualized with seaborn to observe classification performance per activity.

10. Hyperparameter Optimization for SVM (GridSearchCV)

Hyperparameters C and gamma are tuned using GridSearchCV with 5-fold cross-validation to maximize accuracy.

11. Label Encoding for Activity Classes

String activity labels (e.g., “Walking”, “Sitting”) are encoded into integers for model compatibility.

12. XGBoost Model Training and Evaluation

An XGBoost classifier is trained and evaluated on the encoded dataset.
Classification report and confusion matrix are generated for comparison with SVM.

13. Hyperparameter Optimization for XGBoost

A grid search is performed over parameters like n_estimators, max_depth, and learning_rate to optimize the model.

14. Model Performance Comparison (Accuracy & F1 Score)

The performance of SVM and XGBoost is compared side-by-side using Accuracy and F1 Score metrics.

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].bar(results['Model'], results['Accuracy'], color='skyblue', alpha=0.8)
axes[0].set_title("Model Accuracy Comparison")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)

axes[1].bar(results['Model'], results['F1 Score'], color='salmon', alpha=0.8)
axes[1].set_title("Model F1 Score Comparison")
axes[1].set_ylabel("F1 Score")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()


| Model   | Accuracy | F1 Score        |
| ------- | -------- | --------------- |
| SVM     | 0.93     | (varies by run) |
| XGBoost | 0.93     | (varies by run) |

Both models perform comparably, showing strong classification performance across all activity types.
