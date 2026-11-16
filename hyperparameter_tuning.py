from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
# HYPERPARAMETER TUNING
def tune_model(
    preprocessor,
    X_train,
    y_train
):
    print("\nCreating Tuning Pipeline...")
    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "feature_selection",
            SelectFromModel(
                ExtraTreesClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=1
                ),
                threshold="median"
            )
        ),
        (
            "model",
            ExtraTreesClassifier(
                random_state=42,
                n_jobs=1
            )
        )
    ])
    # PARAMETERS TO TUNE
    param_distributions = {
        "model__n_estimators": [100, 150, 200],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    }
    print("\nStarting RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=1,
        verbose=2
    )
    random_search.fit(
        X_train,
        y_train
    )
    print("\nBEST PARAMETERS")
    print(
        random_search.best_params_
    )
    print("\nBEST CROSS VALIDATION F1 SCORE")
    print(
        random_search.best_score_
    )
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_
    return best_model, best_score
