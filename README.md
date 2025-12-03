# Multiple Disease prediction

This project is a Machine Learning–based Multiple Disease Prediction System built using Python, Scikit-learn, Streamlit, and Joblib.
It allows users to input medical parameters and predict the likelihood of:

Liver Disease

Kidney Disease

Parkinson Disease

The goal is to create a simple, user-friendly, and interactive web interface where users can check disease predictions instantly.

**Features**

✔ Predict 3 diseases using trained ML models

✔ Built with Streamlit for a fast and interactive UI

✔ Uses multiple ML models (Logistic Regression, KNN, Decision Tree, Random Forest, Bagging)

✔ Models saved using joblib

✔ Clean UI with disease-specific images

✔ All datasets preprocessed (scaling, encoding, missing value handling)

✔ Supports multiple user inputs dynamically

✔ Fast & offline predictions

**Technologies Used**
| Category             | Tools / Libraries         |
| -------------------- | ------------------------- |
| **Frontend / UI**    | Streamlit                 |
| **Machine Learning** | Scikit-learn              |
| **Data Processing**  | Pandas, NumPy             |
| **Model Saving**     | Joblib                    |
| **Visualization**    | Matplotlib, Seaborn       |
| **File Format**      | Jupyter Notebook (.ipynb) |

**Machine Learning Models Used**

Each disease has multiple ML models:

**Liver Disease Models**

Logistic Regression

KNN

Decision Tree

Random Forest

Bagging Classifier

StandardScaler

**Kidney Disease Models**

Logistic Regression

KNN

Decision Tree

Random Forest

StandardScaler

**Parkinson Disease Models**

Logistic Regression

KNN

Decision Tree

Random Forest

StandardScaler

**How the Application Works**

1. User selects Disease Type

Liver

Kidney

Parkinson

2. User enters required medical parameters

Streamlit automatically generates input fields.

3. Data is passed to the StandardScaler

Scaler ensures model gets correct format.

4. The selected ML model predicts:

🔴 Disease Present or
🟢 No Disease

5. Result is shown with colors and icons for better readability.

**Screenshots**

![Liver Prediction](images/liver.png)
![Kidney Prediction](images/kidney.png)
![Parkinson Prediction](images/parkinson.png)


**Model Training Notebooks**

Liver-	liver.ipynb

Kidney-	kidney.ipynb

Parkinson- parkinson.ipynb

-->Each notebook contains:

✔ Cleaning

✔ Encoding

✔ Missing value handling

✔ Scaling

✔ Train/Test Split

✔ Model training

✔ Accuracy reports

✔ Saving models using joblib

@ Contact: 📧 Email: sneharaje167@gmail.com

🌐 LinkedIn: https://www.linkedin.com/in/sneha-rajendiran-2427651b7
