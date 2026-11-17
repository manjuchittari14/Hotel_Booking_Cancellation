import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.ensemble import ExtraTreesClassifier
from src.preprocessing import create_preprocessor
from src.models import create_model_pipeline
# STEP 1: CREATE FASTAPI APPLICATION
app = FastAPI(
    title="Hotel Booking Cancellation Prediction",
    description="Predict whether a hotel booking will be cancelled",
    version="1.0"
)
BASE_DIR = Path(__file__).resolve().parent
# STEP 2: LOAD AND PREPROCESS DATASET
print("\nLoading Hotel Booking Dataset...")
df = pd.read_csv(BASE_DIR / "data" / "hotel_bookings.csv")
print("Original Dataset Shape:", df.shape)
df = df.drop_duplicates().copy()
df = df.dropna(subset=["is_canceled"]).copy()
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
print("Cleaned Dataset Shape:", df.shape)
# STEP 3: PREPARE FEATURES AND TARGET
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"].astype(int)
if "agent" in X.columns:
    X["agent"] = X["agent"].astype("object")
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()
print("Number of Input Features:", len(X.columns))
print("Input Columns:", X.columns.tolist())
# STEP 4: CREATE AND TRAIN MODEL
preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)
classifier = ExtraTreesClassifier(
    n_estimators=150,
    max_features="sqrt",
    random_state=42,
    n_jobs=1
)
model = create_model_pipeline(
    preprocessor,
    classifier
)
print("\nTraining Hotel Booking Model...")
model.fit(X, y)
print("Model Training Completed Successfully!")
# STEP 5: BOOKING INPUT SCHEMA
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
    country: str = "PRT"
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    deposit_type: str
    agent: float = 0
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    # Include these if they are features in your dataset
    booking_changes: int = 0
    days_in_waiting_list: int = 0
# STEP 6: HOME ROUTE
@app.get("/")
def home():
    return {
        "project": "Hotel Booking Cancellation Prediction",
        "status": "API Running Successfully",
        "target": "is_canceled"
    }
# STEP 7: HTML UI ROUTE
@app.get("/ui")
def home_page():
    html_file = BASE_DIR / "templates" / "index.html"
    if not html_file.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html file not found"
        )
    return FileResponse(
        html_file,
        media_type="text/html"
    )
# STEP 8: PREDICTION ROUTE
@app.post("/predict")
def predict_booking(data: BookingInput):
    try:
        input_data = pd.DataFrame([data.dict()])
        if "agent" in input_data.columns:
            input_data["agent"] = input_data["agent"].astype("object")
        # Check that all model features are supplied
        missing_columns = [
            col for col in X.columns
            if col not in input_data.columns
        ]
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing model input features",
                    "missing_columns": missing_columns
                }
            )
        input_data = input_data[X.columns]
        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0][1]
        )
        result = (
            "Booking Cancelled"
            if prediction == 1
            else "Booking Not Cancelled"
        )
        return {
            "is_canceled": prediction,
            "prediction": result,
            "cancellation_probability": round(
                probability * 100, 2
            ),
            "message": "Prediction completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
# STEP 9: HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model": "Trained and Ready"
    }
=======
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.ensemble import ExtraTreesClassifier
from src.preprocessing import create_preprocessor
from src.models import create_model_pipeline
# STEP 1: CREATE FASTAPI APPLICATION
app = FastAPI(
    title="Hotel Booking Cancellation Prediction",
    description="Predict whether a hotel booking will be cancelled",
    version="1.0"
)
BASE_DIR = Path(__file__).resolve().parent
# STEP 2: LOAD AND PREPROCESS DATASET
print("\nLoading Hotel Booking Dataset...")
df = pd.read_csv(BASE_DIR / "data" / "hotel_bookings.csv")
print("Original Dataset Shape:", df.shape)
df = df.drop_duplicates().copy()
df = df.dropna(subset=["is_canceled"]).copy()
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
print("Cleaned Dataset Shape:", df.shape)
# STEP 3: PREPARE FEATURES AND TARGET
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"].astype(int)
if "agent" in X.columns:
    X["agent"] = X["agent"].astype("object")
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()
print("Number of Input Features:", len(X.columns))
print("Input Columns:", X.columns.tolist())
# STEP 4: CREATE AND TRAIN MODEL
preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)
classifier = ExtraTreesClassifier(
    n_estimators=150,
    max_features="sqrt",
    random_state=42,
    n_jobs=1
)
model = create_model_pipeline(
    preprocessor,
    classifier
)
print("\nTraining Hotel Booking Model...")
model.fit(X, y)
print("Model Training Completed Successfully!")
# STEP 5: BOOKING INPUT SCHEMA
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
    country: str = "PRT"
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    deposit_type: str
    agent: float = 0
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    # Include these if they are features in your dataset
    booking_changes: int = 0
    days_in_waiting_list: int = 0
# STEP 6: HOME ROUTE
@app.get("/")
def home():
    return {
        "project": "Hotel Booking Cancellation Prediction",
        "status": "API Running Successfully",
        "target": "is_canceled"
    }
# STEP 7: HTML UI ROUTE
@app.get("/ui")
def home_page():
    html_file = BASE_DIR / "templates" / "index.html"
    if not html_file.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html file not found"
        )
    return FileResponse(
        html_file,
        media_type="text/html"
    )
# STEP 8: PREDICTION ROUTE
@app.post("/predict")
def predict_booking(data: BookingInput):
    try:
        input_data = pd.DataFrame([data.dict()])
        if "agent" in input_data.columns:
            input_data["agent"] = input_data["agent"].astype("object")
        # Check that all model features are supplied
        missing_columns = [
            col for col in X.columns
            if col not in input_data.columns
        ]
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing model input features",
                    "missing_columns": missing_columns
                }
            )
        input_data = input_data[X.columns]
        prediction = int(model.predict(input_data)[0])
        probability = float(
            model.predict_proba(input_data)[0][1]
        )
        result = (
            "Booking Cancelled"
            if prediction == 1
            else "Booking Not Cancelled"
        )
        return {
            "is_canceled": prediction,
            "prediction": result,
            "cancellation_probability": round(
                probability * 100, 2
            ),
            "message": "Prediction completed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
# STEP 9: HEALTH CHECK
@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model": "Trained and Ready"
    }
