import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from src.models import create_model_pipeline
# STEP 1: CROSS VALIDATION
def perform_cross_validation(
    models,
    preprocessor,
    X_train,
    y_train
):
    results = {}
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1 Score": "f1",
        "ROC AUC": "roc_auc"
    }
    for name, model in models.items():
        print(f"\nCross-validating: {name}")
        pipeline = create_model_pipeline(
            preprocessor,
            model
        )
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )
        results[name] = {
            "Accuracy": scores["test_Accuracy"].mean(),
            "Precision": scores["test_Precision"].mean(),
            "Recall": scores["test_Recall"].mean(),
            "F1 Score": scores["test_F1 Score"].mean(),
            "ROC AUC": scores["test_ROC AUC"].mean()
        }
        print("Accuracy:", results[name]["Accuracy"])
        print("F1 Score:", results[name]["F1 Score"])
    return results
# STEP 2: FINAL MODEL EVALUATION
def evaluate_model(
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):
    # If model is already a full pipeline, use it directly.
    if (
        isinstance(model, Pipeline)
        and "preprocessor" in model.named_steps
    ):
        final_pipeline = model
    else:
        final_pipeline = create_model_pipeline(
            preprocessor,
            model
        )
    print("\nTraining Final Model...")
    final_pipeline.fit(
        X_train,
        y_train
    )
    predictions = final_pipeline.predict(
        X_test
    )
    probabilities = final_pipeline.predict_proba(
        X_test
    )[:, 1]
    print("\nFINAL TEST RESULTS")
    print(
        "Accuracy:",
        accuracy_score(y_test, predictions)
    )
    print(
        "Precision:",
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )
    print(
        "Recall:",
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    )
    print(
        "F1 Score:",
        f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    )
    print(
        "ROC AUC:",
        roc_auc_score(
            y_test,
            probabilities
        )
    )
    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )
    return final_pipeline
