# Spambase Classification Using Machine Learning

## A. Problem Statement

Email spam is a common problem that affects users by filling inboxes with unwanted and potentially harmful messages. Identifying spam emails automatically is important for maintaining secure and organized email communication.

The objective of this project is to build and evaluate machine learning classification models that can distinguish between spam and non-spam emails using the UCI Spambase dataset.

Five classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Their performance is evaluated using Accuracy, ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## B. Dataset Description

The dataset used for this project is the **Spambase Dataset**, obtained from the UCI Machine Learning Repository.

**Dataset Source:** [UCI Spambase Dataset](https://archive.ics.uci.edu/dataset/94/spambase)

The dataset contains information extracted from email messages and is used to classify emails as spam or non-spam.

**Dataset Details:**

- Original number of instances: 4,601
- Number of predictor features: 57
- Target variable: 1 (Spam), 0 (Non-Spam)
- Feature types: Numerical
- Missing values: None

The 57 predictor features consist of:

- 48 word frequency features, representing the percentage of words in an email matching specific words.
- 6 character frequency features, representing the percentage of specified characters in an email.
- 3 capital letter run-length features, describing the average, longest and total length of uninterrupted sequences of capital letters.

**Data Cleaning:**

During data exploration, duplicate records were identified. Records with identical predictor values and conflicting target labels were excluded, and redundant duplicate records were reduced to one representative record.

After cleaning, the dataset contained **4,204 instances, 57 predictor features and one target variable**.

**Preprocessing and Data Splitting:**

- The dataset was divided into 80% training data and 20% testing data.
- Stratified sampling was used to maintain the class distribution in both sets.
- StandardScaler was fitted on the training data and applied to Logistic Regression, KNN and SVM.
- Decision Tree and Gaussian Naive Bayes were trained using the original unscaled features.

**Note:** The Spambase dataset is a historical email dataset. Therefore, the results demonstrate the performance of the selected classification algorithms on this dataset and may not fully represent patterns in modern spam emails.

## C. GitHub Repository Link

**GitHub Repository:**  
https://github.com/Sol-Explorer/spambase-classification

**Live Streamlit Application:**  
https://spambase-classificationgit-dyoduxdmdc8tgpdc6zoupc.streamlit.app/

### Streamlit Application – Usage Instructions

1. Open the live Streamlit application using the link above.
2. Upload the test dataset (`test_data.csv`) using the CSV file uploader. The file is available in the root directory of the GitHub repository.
3. Select one of the five classification models from the dropdown:
   - Logistic Regression
   - Decision Tree
   - KNN
   - Gaussian Naive Bayes
   - SVM
4. View the evaluation results, including Accuracy, ROC-AUC, Precision, Recall, F1 Score, MCC, confusion matrix and classification report.

The application uses the selected pre-trained model and its required preprocessing artifact to generate predictions. It does not retrain the model using the uploaded test data.

## D. Models Used

The following five machine learning classification algorithms were implemented and evaluated using the cleaned Spambase dataset:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics Comparison

The performance of each model was evaluated using the following metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Different hyperparameter values were explored for selected models during experimentation. However, no substantial improvement in overall performance was observed compared with the baseline configurations. Therefore, the baseline configurations were retained for the final comparison of all five classification algorithms.

| **ML Model Name** | **Accuracy** | **ROC-AUC** | **Precision** | **Recall** | **F1 Score** | **MCC** |
| ---------------------- | ------------ | ------- | ------------- | ---------- | ------ | ------- |
| Logistic Regression    | 0.9239       | 0.9721  | 0.9045        | 0.9045     | 0.9045 | 0.8412  |
| Decision Tree          | 0.9227       | 0.9186  | 0.9066        | 0.8985     | 0.9025 | 0.8385  |
| KNN                    | 0.9168       | 0.9654  | 0.9027        | 0.8866     | 0.8946 | 0.8259  |
| Naive Bayes (Gaussian) | 0.8252       | 0.9498  | 0.7035        | 0.9701     | 0.8156 | 0.6881  |
| Support Vector Machine | 0.9310       | 0.9766  | 0.9184        | 0.9075     | 0.9129 | 0.8559  |

### Observations on Model Performance

| **ML Model Name** | **Observation about model performance** |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Achieved an accuracy of 92.39% and ROC-AUC of 0.9721, indicating strong overall classification performance. Its accuracy was very close to Decision Tree, while its ROC-AUC was considerably higher (0.9721 vs. 0.9186), suggesting better ability to distinguish between spam and non-spam emails across different classification thresholds. It also achieved a higher F1 Score than Decision Tree and KNN. |
| **Decision Tree** | Achieved an accuracy of 92.27%, which was almost identical to Logistic Regression (92.39%). Its precision was slightly higher than Logistic Regression, but its recall and F1 Score were marginally lower. However, its ROC-AUC of 0.9186 was the lowest among the five models, indicating comparatively weaker class discrimination across different thresholds. |
| **K-Nearest Neighbors (KNN)** | Achieved an accuracy of 91.68% and ROC-AUC of 0.9654, showing good classification performance. Its precision was close to Logistic Regression, but its recall was lower (0.8866 vs. 0.9045), meaning it missed more spam emails. Consequently, its F1 Score was also slightly lower than Logistic Regression and Decision Tree. |
| **Gaussian Naive Bayes** | Achieved the highest recall (0.9701) among all five models, identifying 325 out of 335 spam emails. However, its precision was the lowest (0.7035), with 137 non-spam emails incorrectly classified as spam. This indicates that while the model identified most spam emails, it also generated considerably more false positives. Its accuracy and MCC were also the lowest among the models. |
| **Support Vector Machine (SVM)** | Achieved the highest accuracy (93.10%), ROC-AUC (0.9766), precision (0.9184), F1 Score (0.9129) and MCC (0.8559). Compared with Logistic Regression, it achieved improvements across all six metrics, although the differences were relatively small for some metrics. It also produced fewer false positives than the other models, making it the strongest overall performer in this comparison. |
| **Overall Winner for the Dataset** | **Support Vector Machine (SVM)** performed best overall, achieving the highest scores in five out of six evaluation metrics. Although Gaussian Naive Bayes achieved the highest recall, its lower precision resulted in more false positives. Considering the overall balance of classification performance across the evaluation metrics, SVM was the best-performing model for this dataset. |