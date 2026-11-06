from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
# FEATURE SELECTION + MODEL PIPELINE
def create_model_pipeline(preprocessor, classifier):
    pipeline = Pipeline([
        # Step 1: Preprocessing
        ("preprocessor", preprocessor),
        # Step 2: Feature Selection
        ("feature_selection", SelectFromModel(
            ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=1
            ),
            threshold="median"
        )),
        # Step 3: Classification Model
        ("model", classifier)
    ])
    return pipeline
# BASE CLASSIFICATION MODELS
def get_base_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            random_state=42
        )
    }
    return models
# ENSEMBLE CLASSIFICATION MODELS
def get_ensemble_models():
    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )
    models = {
        "Random Forest": random_forest
    }
    return models
