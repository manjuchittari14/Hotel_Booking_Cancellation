import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
def load_data():
    df = pd.read_csv(r"C:\Users\sanne\OneDrive\Pictures\Hotel _booking_cancellation\data\hotel_bookings.csv")
    print("Original Dataset Shape:", df.shape)
    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()
    print("Cleaned Dataset Shape:", df.shape)
    return df
def preprocess_data(df):
    # Remove rows where target is missing
    df = df.dropna(subset=["is_canceled"]).copy()
    # Remove leakage and unsuitable columns
    columns_to_drop = [
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "company"
    ]
    df.drop(
        columns=columns_to_drop,
        errors="ignore",
        inplace=True
    )
    # Agent is an ID, so treat it as categorical
    if "agent" in df.columns:
        df["agent"] = df["agent"].astype("object")
    # Separate input and target
    X = df.drop("is_canceled", axis=1)
    y = df["is_canceled"].astype(int)
    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    print("\nInput Features:", X.shape[1])
    print("Numerical Features:", len(numerical_columns))
    print("Categorical Features:", len(categorical_columns))
    print("\nTarget Distribution:")
    print(y.value_counts())
    return X, y, numerical_columns, categorical_columns
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)
    return X_train, X_test, y_train, y_test
def create_preprocessor(
    numerical_columns,
    categorical_columns
):
    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )
    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )
    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )
    return preprocessor
