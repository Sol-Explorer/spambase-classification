
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Spambase Classification")
st.write("Compare five machine learning models for spam detection.")

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"]
)

model_name = st.selectbox(
    "Select a classification model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "SVM"
    ]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.write("Uploaded test data:")
    st.dataframe(test_data.head())

    # Load the selected trained model
    if model_name == "Logistic Regression":
        model = joblib.load("model/logistic_regression.joblib")

    elif model_name == "Decision Tree":
        model = joblib.load("model/decision_tree.joblib")

    elif model_name == "KNN":
        model = joblib.load("model/knn.joblib")

    elif model_name == "Gaussian Naive Bayes":
        model = joblib.load("model/naive_bayes.joblib")

    elif model_name == "SVM":
        model = joblib.load("model/svm.joblib")

    # Separate features and target
    X = test_data.drop("target", axis=1).to_numpy()
    y = test_data["target"]

    # Apply the same preprocessing used during training
    if model_name in ["Logistic Regression", "KNN", "SVM"]:
        scaler = joblib.load("model/scaler.joblib")
        X_input = scaler.transform(X)
    else:
        X_input = X

    # Generate predictions and probabilities
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    mcc = matthews_corrcoef(y, predictions)

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "ROC-AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ],
        "Score": [
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc
        ]
    })

    st.dataframe(metrics, hide_index=True)

    # Display confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    confusion_data = pd.DataFrame(
        cm,
        index=["Actual Non-Spam", "Actual Spam"],
        columns=["Predicted Non-Spam", "Predicted Spam"]
    )

    st.dataframe(confusion_data)
