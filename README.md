# Hotel Booking Cancellation Prediction
## Project Overview
This project predicts whether a hotel booking will be canceled using Machine Learning classification algorithms.
The goal is to help hotels understand cancellation patterns and improve booking management.
## Objectives
- Predict booking cancellations.
- Analyze hotel booking data.
- Compare multiple classification models.
- Identify the best-performing model.
- Improve model performance using ML techniques.
## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- VS Code
## Dataset
The project uses the Hotel Booking Demand dataset.
Target variable: `is_canceled`
- 0 = Booking not canceled
- 1 = Booking canceled
## Machine Learning Models
### Base Models
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
### Ensemble Models
- Bagging
- AdaBoost
- Gradient Boosting
- Voting Classifier
- Stacking Classifier
## Techniques Implemented
- Data preprocessing
- Feature selection
- Train-test split
- Cross-validation
- Hyperparameter tuning
- Machine Learning pipelines
- Model comparison
- Classification evaluation
## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
## Project Structure
```text
Hotel-Booking-Cancellation/
├── data/
├── src/
├── outputs/
├── main.py
├── predict.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```
## Installation
Install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Run the Project
```bash
python main.py
``
## Run Streamlit Application
```bash
python -m streamlit run streamlit_app.py
```
## Conclusion
This project uses Machine Learning classification techniques to predict hotel booking cancellations and compares different models to identify effective prediction methods.
## Author
AIML Machine Learning Project
