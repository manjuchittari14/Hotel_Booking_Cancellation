import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
# 1. CREATE FASTAPI APPLICATION
app = FastAPI(
    title="Hotel Booking Cancellation Prediction",
    description="Machine Learning API to predict hotel booking cancellations",
    version="1.0"
)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"
# 2. DEFINE INPUT DATA
class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float = 0
    babies: int = 0
    meal: str
    country: Optional[str] = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    deposit_type: str
    agent: Optional[str] = None
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    booking_changes: int = 0
    days_in_waiting_list: int = 0
# 3. LOAD AND PREPROCESS DATASET
def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found: " + str(DATA_PATH)
        )
    df = pd.read_csv(DATA_PATH)
    # Remove rows with missing target
    df = df.dropna(subset=["is_canceled"])
    # Remove data leakage columns
    columns_to_drop = [
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "company"
    ]
    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )
    # Treat agent as categorical data
    df["agent"] = df["agent"].astype("object")
    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"].astype(int)
    # Convert agent values to strings consistently
    X["agent"] = X["agent"].apply(
        lambda value: str(value) if pd.notna(value) else None
    )
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return X, y, numerical_columns, categorical_columns
# 4. TRAIN MACHINE LEARNING MODEL
X, y, numerical_columns, categorical_columns = (
    load_training_data()
)
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ExtraTreesClassifier(
        n_estimators=150,
        max_features="sqrt",
        random_state=42,
        n_jobs=1
    ))
])
model.fit(X, y)
print("Dataset Loaded Successfully!")
print("Training Data Shape:", X.shape)
print("Model Training Completed Successfully!")
# 5. HOME ROUTE
@app.get("/")
def home():
    return {
        "project": "Hotel Booking Cancellation Prediction",
        "status": "API Running Successfully",
        "target": "is_canceled",
        "classes": {
            "0": "Booking Not Cancelled",
            "1": "Booking Cancelled"
        }
    }
# 6. HTML USER INTERFACE
@app.get("/ui")
def home_page():
    html_path = BASE_DIR / "templates" / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found inside templates folder"
        )
    return FileResponse(html_path)
# 7. PREDICTION ROUTE
@app.post("/predict")
def predict_booking(data: BookingInput):
    try:
        # Convert incoming data into DataFrame
        input_data = pd.DataFrame([data.dict()])
        # Match the training feature format
        input_data["agent"] = input_data["agent"].apply(
            lambda value: str(value) if pd.notna(value) else None
        )
        # Ensure every expected feature is supplied
        missing_columns = [
            column for column in X.columns
            if column not in input_data.columns
        ]
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing input features",
                    "missing_columns": missing_columns
                }
            )
        input_data = input_data[X.columns]
        # Predict cancellation
        prediction = int(model.predict(input_data)[0])
        # Calculate cancellation probability
        probabilities = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(1)
        cancellation_probability = float(
            probabilities[class_index]
        )
        if prediction == 1:
            message = "This booking is likely to be cancelled."
            status = "Booking Cancelled"
        else:
            message = "This booking is likely not to be cancelled."
            status = "Booking Not Cancelled"
        return {
            "prediction": prediction,
            "status": status,
            "cancellation_probability": round(
                cancellation_probability * 100, 2
            ),
            "message": message
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
# 8. HEALTH CHECK
@app.get("/health")
def health_check():
    return {
        "status": "Healthy",
        "model": "Extra Trees Classifier",
        "message": "Prediction API is ready"
    }
