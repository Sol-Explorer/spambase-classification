# Spambase Classification

## 1. Project Overview

This project implements and compares five machine learning classification algorithms to identify whether an email is spam or non-spam using the UCI Spambase dataset.

The project also includes a Streamlit application that allows users to upload test data, select a classification model and view its evaluation results.

## 2. Problem Statement

Email spam detection is a binary classification problem. The objective is to classify emails into two categories:

- **0 – Non-Spam**
- **1 – Spam**

Five classification algorithms are implemented and evaluated on the same dataset to compare their performance.

## 3. Dataset Description

**Dataset:** Spambase  
**Source:** UCI Machine Learning Repository  
**Dataset link:** https://archive.ics.uci.edu/dataset/94/spambase

The original dataset contains:

- 4,601 instances
- 57 numerical predictor features
- 1 binary target variable

The features represent word frequencies, character frequencies and capital-letter sequences in emails.

During data preparation, duplicate records with identical predictor values were examined. Redundant duplicates and records with conflicting target labels were excluded, resulting in a cleaned dataset of 4,204 instances.

## 4. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The dataset was divided into 80% training data and 20% testing data using stratified sampling.

Feature scaling using StandardScaler was applied to Logistic Regression, KNN and SVM.

## 5. Evaluation Metrics

Each model was evaluated using the following six metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9239 | 0.9721 | 0.9045 | 0.9045 | 0.9045 | 0.8412 |
| Decision Tree | 0.9227 | 0.9186 | 0.9066 | 0.8985 | 0.9025 | 0.8385 |
| KNN | 0.9168 | 0.9654 | 0.9027 | 0.8866 | 0.8946 | 0.8259 |
| Gaussian Naive Bayes | 0.8252 | 0.9498 | 0.7035 | 0.9701 | 0.8156 | 0.6881 |
| SVM | 0.9310 | 0.9766 | 0.9184 | 0.9075 | 0.9129 | 0.8559 |

### Observations

- **Logistic Regression:** Achieved good overall performance with an accuracy of 92.39% and ROC-AUC of 0.9721.
- **Decision Tree:** Achieved 92.27% accuracy, with performance comparable to Logistic Regression.
- **KNN:** Achieved 91.68% accuracy and a ROC-AUC of 0.9654.
- **Gaussian Naive Bayes:** Achieved the highest recall of 97.01%, but its precision was lower at 70.35%, indicating more false-positive predictions.
- **SVM:** Achieved the highest accuracy (93.10%), ROC-AUC (0.9766), F1 score (0.9129) and MCC (0.8559) among the five models.

### Overall Result

SVM demonstrated the best overall performance on this dataset based on the evaluation metrics.

## 6. Streamlit Application

The project includes a Streamlit application with the following features:

- Upload test data in CSV format.
- Select one of the five trained classification models.
- View the six evaluation metrics.
- View the confusion matrix.

The application loads the previously trained and saved models. It does not perform model training again.

## 7. Project Structure

```text
spambase-classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── data/
│   ├── spambase.data
│   ├── spambase.names
│   └── spambase.DOCUMENTATION
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── scaler.joblib
│
└── source_code/
    └── spambase_classification.ipynb
```

## 8. How to Run the Application

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload the test CSV file, select a classification model and view its evaluation results.

## 9. GitHub Repository

Repository: https://github.com/Sol-Explorer/spambase-classification.git
