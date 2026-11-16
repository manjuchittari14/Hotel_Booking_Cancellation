import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# PAGE CONFIGURATION
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)
# TITLE
st.title("🏨 Hotel Booking Cancellation Prediction")
st.subheader("Machine Learning Classification Dashboard")
st.write(
    "This project predicts whether a hotel booking "
    "will be cancelled using Machine Learning."
)
st.divider()
# PROJECT OVERVIEW
st.header("📊 Project Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Dataset Rows", "119,390")
col2.metric("Input Features", "27")
col3.metric("Training Samples", "95,512")
col4.metric("Testing Samples", "23,878")
st.divider()
# FINAL MODEL PERFORMANCE
st.header("🏆 Final Model Performance")
st.success("Selected Best Model: Random Forest Classifier")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", "89.04%")
col2.metric("Precision", "88.88%")
col3.metric("Recall", "80.49%")
col4.metric("F1 Score", "84.47%")
col5.metric("ROC AUC", "95.67%")
st.divider()
# MODEL COMPARISON
st.header("📈 Model Comparison")
model_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "Voting Classifier"
    ],
    "Accuracy": [
        82.78, 86.01, 88.57, 87.76, 88.50
    ],
    "Precision": [
        81.02, 82.47, 88.77, 86.54, 88.72
    ],
    "Recall": [
        69.87, 79.08, 79.17, 79.29, 78.99
    ],
    "F1 Score": [
        75.03, 80.72, 83.69, 82.76, 83.58
    ],
    "ROC AUC": [
        90.75, 92.71, 95.36, 94.69, 95.30
    ]
}
df_models = pd.DataFrame(model_data)
st.dataframe(
    df_models,
    use_container_width=True,
    hide_index=True
)
# ACCURACY GRAPH
st.subheader("Model Accuracy Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    df_models["Model"],
    df_models["Accuracy"]
)
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Machine Learning Models")
ax.set_title("Classification Model Accuracy")
ax.set_ylim(0, 100)
plt.xticks(rotation=25)
st.pyplot(fig)
# CROSS VALIDATION
st.header("🔄 Cross Validation Results")
st.write("Cross-validation was performed to compare model performance.")
cv_data = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "Voting Classifier"
    ],
    "CV Accuracy": [
        82.78, 86.01, 88.57, 87.76, 88.50
    ],
    "CV F1 Score": [
        75.03, 80.72, 83.69, 82.76, 83.58
    ],
    "CV ROC AUC": [
        90.75, 92.71, 95.36, 94.69, 95.30
    ]
}
st.dataframe(
    pd.DataFrame(cv_data),
    use_container_width=True,
    hide_index=True
)
# HYPERPARAMETER TUNING
st.header("⚙️ Hyperparameter Tuning")
st.write("Method: Randomized/Grid Search with 3-Fold CV")
st.write("Best Parameters:")
parameters = {
    "Number of Estimators": 150,
    "Maximum Depth": "None",
    "Minimum Samples Split": 2,
    "Maximum Features": "sqrt"
}
st.json(parameters)
st.metric(
    "Best Cross Validation F1 Score",
    "82.95%"
)
# FINAL CLASSIFICATION REPORT
st.header("📋 Classification Report")
report_data = {
    "Class": [
        "Not Cancelled (0)",
        "Cancelled (1)"
    ],
    "Precision": [
        0.89, 0.89
    ],
    "Recall": [
        0.94, 0.80
    ],
    "F1 Score": [
        0.92, 0.84
    ],
    "Support": [
        15033, 8845
    ]
}
st.dataframe(
    pd.DataFrame(report_data),
    use_container_width=True,
    hide_index=True
)
# FINAL CONCLUSION
st.header("📝 Project Conclusion")
st.write("""
The Hotel Booking Cancellation Prediction project was
implemented using multiple Machine Learning classification
algorithms.
Random Forest achieved the best final accuracy of 89.04%.
The model can help hotels identify cancellation risks and
improve booking management.
The Random Forest model was selected based on its overall
performance across accuracy, precision, recall, F1 score,
and ROC AUC.
""")
st.success("Hotel Booking Cancellation Dashboard Completed!")
st.caption("Developed using Python, Scikit-learn, Pandas and Streamlit")