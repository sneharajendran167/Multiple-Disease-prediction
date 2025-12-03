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

1.User selects Disease Type

Liver

Kidney

Parkinson

2.User enters required medical parameters

Streamlit automatically generates input fields.

3.Data is passed to the StandardScaler

Scaler ensures model gets correct format.

4.The selected ML model predicts:

🔴 Disease Present or

🟢 No Disease

5.Result is shown with colors and icons for better readability.

**Screenshots**
<img width="1920" height="1080" alt="kidney" src="https://github.com/user-attachments/assets/069b87dd-2537-469d-9828-aff614e076e2" />

<img width="1920" height="1080" alt="parkiston" src="https://github.com/user-attachments/assets/d4f80a14-215b-449a-bf65-caf7510bed86" />

<img width="1920" height="1080" alt="liver" src="https://github.com/user-attachments/assets/dc5164dd-d014-4fcb-8578-d4aaf5bab5c4" />

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

**Conclusion**

The Multiple Disease Prediction System successfully demonstrates how Machine Learning can support early detection of critical health conditions such as Liver Disease, Kidney Disease, and Parkinson’s Disease. By integrating multiple ML models, real-world medical datasets, and an interactive Streamlit interface, this project provides a fast, reliable, and user-friendly solution for health risk assessment.

The system allows users to enter medical parameters and instantly receive predictions using well-trained models, helping in early diagnosis, preventive care, and clinical decision support.

Overall, this project proves that combining data science, machine learning, and healthcare domain knowledge can create impactful tools that contribute to improving human health and well-being.

@ Contact: 📧 Email: sneharaje167@gmail.com

🌐 LinkedIn: https://www.linkedin.com/in/sneha-rajendiran-2427651b7
