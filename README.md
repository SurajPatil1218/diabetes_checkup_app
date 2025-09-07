# diabetes_checkup_app
Diabetes_checkup_app
🩺 Diabetes Checkup App

This is a Streamlit-based web application that predicts whether a person is diabetic or not using Random Forest Classifier.
The app is interactive and allows users to input their health parameters through sidebar sliders to get instant predictions.

## Features
Uploads and processes diabetes dataset (diabetes.csv).
Displays statistical summary of the dataset.
Provides data visualization using bar charts. Trains a Random Forest Classifier model on the dataset. 
Shows model accuracy.
Allows users to input health parameters (Pregnancies, Glucose, Blood Pressure, etc.) to get a personal health report. 
Outputs whether the user is Healthy or Not Healthy.

## Technologies Used
Python

Streamlit (for web UI)

Pandas (for data handling)

Scikit-learn (for ML model and accuracy)

PIL (for image handling if extended)

## Model Information
Explain the Random Forest Classifier and why you chose it.

Mention that you split data into training and testing sets (80/20).

Example:
Random Forest is an ensemble model that performs well on classification problems by combining multiple decision trees. It provides better accuracy and reduces overfitting.

Example Accuracy: ~78%

Why Random Forest? It is an ensemble method that combines multiple decision trees, reducing overfitting and improving prediction accuracy.
