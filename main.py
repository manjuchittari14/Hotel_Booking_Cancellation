import os
import pandas as pd
from src.preprocessing import (
    preprocess_data,
    split_data,
    create_preprocessor
)
from src.models import (
    get_base_models,
    get_ensemble_models,
    create_model_pipeline
)
from src.evaluation import (
    perform_cross_validation,
    evaluate_model
)
from src.hyperparameter_tuning import tune_model
# CREATE OUTPUT FOLDER
os.makedirs("outputs", exist_ok=True)
# STEP 1: LOAD DATASET
print("\nSTEP 1: LOAD DATASET")
df = pd.read_csv(
    r"C:\Users\sanne\OneDrive\Pictures\Hotel _booking_cancellation\data\hotel_bookings.csv"
)
print("\nDATASET INFORMATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Dataset Shape:", df.shape)
# STEP 2: PREPROCESSING
print("\nSTEP 2: PREPROCESSING")
X, y, numerical_columns, categorical_columns = preprocess_data(df)
# STEP 3: TRAIN TEST SPLIT
print("\nSTEP 3: TRAIN TEST SPLIT")
X_train, X_test, y_train, y_test = split_data(X, y)
# STEP 4: CREATE PREPROCESSOR
print("\nSTEP 4: CREATING PREPROCESSOR")
preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)
# STEP 4.1: FEATURE SELECTION PIPELINE
print("\nSTEP 4.1: FEATURE SELECTION")
feature_selection_pipeline = create_model_pipeline(
    preprocessor,
    get_base_models()["Logistic Regression"]
)
print("Feature Selection Pipeline Created")
# STEP 5: BASE MODELS
print("\nSTEP 5: BASE MODELS")
base_models = get_base_models()
# STEP 6: ENSEMBLE MODELS
print("\nSTEP 6: ENSEMBLE MODELS")
ensemble_models = get_ensemble_models()
# COMBINE ALL MODELS
all_models = {
    **base_models,
    **ensemble_models
}
# STEP 7: CROSS VALIDATION
print("\nSTEP 7: CROSS VALIDATION")
cv_results = perform_cross_validation(
    all_models,
    preprocessor,
    X_train,
    y_train
)
# MODEL COMPARISON
results_df = pd.DataFrame(cv_results).T
print("\nMODEL COMPARISON")
print(results_df)
results_df.to_csv(
    "outputs/model_comparison.csv"
)
# STEP 8: HYPERPARAMETER TUNING
print("\nSTEP 8: HYPERPARAMETER TUNING")
best_model, best_score = tune_model(
    preprocessor,
    X_train,
    y_train
)
# ADD TUNED MODEL RESULT
results_df.loc["Tuned Extra Trees"] = {
    "Accuracy": float("nan"),
    "Precision": float("nan"),
    "Recall": float("nan"),
    "F1 Score": best_score,
    "ROC AUC": float("nan")
}
results_df.to_csv(
    "outputs/model_comparison.csv"
)
# STEP 9: SELECT BEST MODEL USING CV F1
print("\nSTEP 9: SELECTING BEST MODEL")
best_model_name = results_df["F1 Score"].idxmax()
if best_model_name == "Tuned Extra Trees":
    selected_model = best_model
else:
    selected_model = all_models[best_model_name]
print("\nSelected Model:", best_model_name)
# STEP 10: FINAL TEST EVALUATION
print("\nSTEP 10: FINAL EVALUATION")
final_pipeline = evaluate_model(
    selected_model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
)
# STEP 11: SAMPLE PREDICTION
print("\nSTEP 11: SAMPLE PREDICTION")
sample_customer = X_test.iloc[[0]]
prediction = final_pipeline.predict(
    sample_customer
)
probability = final_pipeline.predict_proba(
    sample_customer
)[0][1]
if prediction[0] == 1:
    print("Booking Prediction: CANCELLED")
else:
    print("Booking Prediction: NOT CANCELLED")

print(
    "Cancellation Probability:",
    round(probability * 100, 2),
    "%"
)
print("\nPROJECT EXECUTION COMPLETED")
