# 🏨 Hotel Booking Cancellation Prediction
An end-to-end Machine Learning project that predicts whether a hotel booking will be cancelled or not using customer booking details and historical hotel booking data.
## 🚀 Project Overview
The Hotel Booking Cancellation Prediction System uses classification algorithms to identify the likelihood of a hotel reservation being cancelled. It helps hotels understand booking patterns, reduce cancellation risks, and improve reservation management.
The project uses data preprocessing, feature selection, machine learning algorithms, cross-validation, hyperparameter tuning, and model evaluation to build a predictive system.
## 🎯 Key Features
- **Booking Cancellation Prediction:** Predicts whether a hotel reservation will be cancelled or not.
- **Data Preprocessing:** Handles missing values, categorical variables, and numerical features.
- **Exploratory Data Analysis (EDA):** Analyzes booking trends, customer behavior, and cancellation patterns.
- **Feature Selection:** Identifies important features that influence booking cancellations.
- **Machine Learning Models:** Implements multiple classification algorithms.
- **Ensemble Learning:** Uses Random Forest and Extra Trees algorithms to improve prediction performance.
- **Cross-Validation:** Evaluates model performance across multiple data splits.
- **Hyperparameter Tuning:** Optimizes model parameters for better performance.
- **Model Evaluation:** Evaluates models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Web Application:** Provides a user interface to enter booking details and obtain cancellation predictions.
- **FastAPI Integration:** Provides REST API endpoints for real-time predictions.
## 💻 Technology Stack
| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Classification | Logistic Regression, Decision Tree, Random Forest, Extra Trees |
| Ensemble Learning | Random Forest, Extra Trees |
| Model Optimization | RandomizedSearchCV / Hyperparameter Tuning |
| Feature Selection | Extra Trees Feature Importance |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Development Environment | VS Code |
| Version Control | Git, GitHub |
## 📊 Dataset Information
- **Dataset:** Hotel Booking Demand Dataset
- **Source:** Kaggle
- **Original Dataset Size:** 119,390 rows and 32 columns
- **Target Variable:** `is_canceled`
### Target Variable
| Value | Description |
|---|---|
| 0 | Booking Not Cancelled |
| 1 | Booking Cancelled |
The dataset contains hotel reservation information, including:
- Hotel type
- Lead time
- Arrival date
- Number of adults and children
- Number of weekend and weekday nights
- Meal type
- Country
- Market segment
- Distribution channel
- Customer type
- Previous cancellations
- Deposit type
- Booking changes
- Special requests
## 🤖 Machine Learning Algorithms
The project explores the following classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Extra Trees Classifier
### Model Development Process
1. Load the hotel booking dataset.
2. Perform exploratory data analysis.
3. Handle missing values.
4. Remove unnecessary and target-leakage columns.
5. Separate numerical and categorical features.
6. Apply data preprocessing.
7. Split the dataset into training and testing sets.
8. Train multiple classification models.
9. Perform cross-validation.
10. Apply feature selection.
11. Perform hyperparameter tuning.
12. Evaluate and compare model performance.
13. Integrate the trained prediction workflow with FastAPI.
## 📈 Model Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report
These metrics help compare model performance and identify the most suitable model for predicting hotel booking cancellations.
## 📁 Project Structure
```text
Hotel_Booking_Cancellation/
│
├── data/
│   └── hotel_bookings.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── hyperparameter_tuning.py
│
├── templates/
│   └── index.html
│
├── api.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
*Note: Update the structure if your actual project contains additional files or folders.*
## ⚙️ Installation and Local Setup
### 1. Clone the Repository
```bash
git clone https://github.com/manjuchittari14/Hotel_Booking_Cancellation.git
```
### 2. Navigate to the Project Directory
```bash
cd Hotel_Booking_Cancellation
```
### 3. Create a Virtual Environment
```bash
python -m venv venv
```
### 4. Activate the Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```
**Mac/Linux:**
```bash
source venv/bin/activate
```
### 5. Install Dependencies
```bash
pip install -r requirements.txt
```
### 6. Run the Machine Learning Project
```bash
python main.py
```
### 7. Start the FastAPI Application
```bash
python -m uvicorn api:app --reload
```
### 8. Access the Application
Open your browser:
- **API Home:** http://127.0.0.1:8000/
- **Swagger Documentation:** http://127.0.0.1:8000/docs
- **Prediction Interface:** http://127.0.0.1:8000/ui
## 🔮 Future Enhancements
- Improve prediction performance through additional feature engineering.
- Add interactive visualizations and analytics dashboards.
- Integrate additional machine learning algorithms.
- Implement user authentication.
- Store prediction history in a database.
- Add real-time booking analytics.
- Improve the user interface and mobile responsiveness.
## 📌 Conclusion
The Hotel Booking Cancellation Prediction project demonstrates the practical application of Machine Learning in the hospitality industry. By analyzing historical booking information and applying classification techniques, the system predicts potential booking cancellations and supports better hotel reservation management.

This project provides practical experience in data preprocessing, exploratory data analysis, classification algorithms, feature selection, cross-validation, hyperparameter tuning, model evaluation, and FastAPI integration.
