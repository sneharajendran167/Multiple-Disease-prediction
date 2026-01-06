import streamlit as st 
import pandas as pd
import numpy as np
import joblib 

# Load models and scalers
liver_models = {
    "Logistic Regression": joblib.load("liver_lr.joblib"),
    "KNN": joblib.load("liver_knn.joblib"),
    "Bagging": joblib.load("liver_bg.joblib"),
    "Decision Tree": joblib.load("liver_dt.joblib"),
    "Random Forest": joblib.load("liver_rf.joblib")
}
liver_scaler = joblib.load("liver_scaler.joblib")

kidney_models = {
    "Logistic Regression": joblib.load("kidney_lr2.joblib"),
    "KNN": joblib.load("kidney_knn2.joblib"),
    "Decision Tree": joblib.load("kidney_dtc2.joblib"),
    "Random Forest": joblib.load("kidney_rfc2.joblib")
}
kidney_scaler = joblib.load("kidney_scaler.joblib")

parkiston_models = {
    "Logistic Regression": joblib.load("parkinson_lr.joblib"),
    "KNN": joblib.load("parkinson_knn.joblib"),
    "Decision Tree": joblib.load("parkinson_dt.joblib"),
    "Random Forest": joblib.load("parkinson_rf.joblib")
}
parkiston_scaler = joblib.load("parkinson_scaler.joblib")

#Feature lists used in training
liver_features = ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase',
                  'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens',
                  'Albumin','Albumin_and_Globulin_Ratio']

kidney_features = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot',
                   'hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']

parkinson_features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP',
                      'MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
                      'RPDE','DFA','spread1','spread2','D2','PPE']

# prediction function
def predict_disease(model, scaler, data, feature_list):

    df = pd.DataFrame([data], columns=feature_list)
    data_scaled = scaler.transform(df)
    prediction = model.predict(data_scaled)
    return prediction[0] 

# Streamlit App
st.sidebar.title("Health Prediction System")
page = st.sidebar.radio("Select Disease", 
                        ["🫁 Liver Disease", "🫘 Kidney Disease", "🧠 Parkinson Disease"])

#liver disease page
if page == "🫁 Liver Disease":
    st.title("Liver Disease Prediction")
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total_Bilirubin",min_value=0.0, max_value=75.0, value=0.6, step=0.1)
    direct_bilirubin = st.number_input("Direct_Bilirubin", min_value=0.0, max_value=20.0, value=0.2, step=0.1)
    alkaline_phosphotase = st.number_input("Alkaline_Phosphotase",min_value=0, max_value=2500, value=120, step=10)
    alamine_amino = st.number_input("Alamine_Aminotransferase",min_value=0, max_value=2000, value=20, step=5)
    aspartate_amino = st.number_input("Aspartate_Aminotransferase",min_value=0, max_value=2000, value=22, step=5)
    total_proteins = st.number_input("Total_Protiens",min_value=2.0, max_value=10.0, value=5.8, step=0.1)
    albumin = st.number_input("Albumin",min_value=1.0, max_value=6.0, value=3.4, step=0.1)
    ratio = st.number_input("Albumin_and_Globulin_Ratio",min_value=0.0, max_value=3.0, value=0.9, step=0.1)

    gender_val = 1 if gender == "Male" else 0
    model_name = st.selectbox("Select Model", list(liver_models.keys()))

    if st.button("Predict Liver Disease"):
        data = [age, gender_val, total_bilirubin, direct_bilirubin,
                alkaline_phosphotase, alamine_amino, aspartate_amino,
                total_proteins, albumin, ratio]
        result = predict_disease(liver_models[model_name], liver_scaler, data, liver_features)

        if result == 1:
            st.error("The person is likely to have Liver Disease.")
        else:
            st.success("The person is unlikely to have Liver Disease.")

#kidney disease page
elif page == "🫘 Kidney Disease":
    st.title("Kidney Disease Prediction")
    col1, col2 = st.columns(2)

    with col1:
            age = st.number_input("Age",min_value=1, max_value=100, value=25)
            bp = st.number_input("Blood Pressure",min_value=50, max_value=180, value=80, step=1)
            sg = st.number_input("Specific Gravity",min_value=1.000, max_value=1.030, value=1.010, step=0.001)
            al = st.number_input("Albumin",min_value=0, max_value=5, value=0, step=1)
            su = st.number_input("Sugar",min_value=0, max_value=5, value=0, step=1)
            rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
            pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
            ba = st.selectbox("Bacteria", ["Present", "Not Present"])
            bgr = st.number_input("Blood Glucose Random",min_value=50, max_value=500, value=100, step=5)

    with col2:
            bu = st.number_input("Blood Urea",min_value=10, max_value=300, value=30, step=5)
            sc = st.number_input("Serum Creatinine",min_value=0.1, max_value=15.0, value=0.9, step=0.1)
            sod = st.number_input("Sodium",min_value=110, max_value=160, value=135, step=1)
            pot = st.number_input("Potassium",min_value=2.5, max_value=7.5, value=4.0, step=0.1)
            hemo = st.number_input("Hemoglobin",min_value=3.0, max_value=18.0, value=13.0, step=0.1)
            pcv = st.number_input("Packed Cell Volume",min_value=10, max_value=60, value=40, step=1)
            wc = st.number_input("White Blood Cell Count",min_value=3000, max_value=20000, value=7000, step=500)
            rc = st.number_input("Red Blood Cell Count",min_value=2.0, max_value=6.5, value=4.5, step=0.1)
            htn = st.selectbox("Hypertension", ["Yes", "No"])
            dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
            cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
            appet = st.selectbox("Appetite", ["Good", "Poor"])
            pe = st.selectbox("Pedal Edema", ["Yes", "No"])
            ane = st.selectbox("Anemia", ["Yes", "No"])
    # Encode categorical variables
    rbc = 1 if rbc == "Normal" else 0
    pc = 1 if pc == "Normal" else 0
    pcc = 1 if pcc == "Present" else 0
    ba = 1 if ba == "Present" else 0
    htn = 1 if htn == "Yes" else 0
    dm = 1 if dm == "Yes" else 0
    cad = 1 if cad == "Yes" else 0
    appet = 1 if appet == "Good" else 0
    pe = 1 if pe == "Yes" else 0
    ane = 1 if ane == "Yes" else 0

    model_name = st.selectbox("Select Model", list(kidney_models.keys()))

    if st.button("Predict Kidney Disease"):
        data = [age, bp, sg, al, su, rbc, pc, pcc, ba,
                bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                htn, dm, cad, appet, pe, ane]
        result = predict_disease(kidney_models[model_name], kidney_scaler, data, kidney_features)

        if result == 1:
            st.error("Positive: Chronic Kidney Disease Detected")
        else:
            st.success("Negative: No Chronic Kidney Disease")

#parkinson disease page
elif page == "🧠 Parkinson Disease":
    st.title("Parkinson Disease Prediction")
    col1, col2 = st.columns(2)
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)",min_value=80.0, max_value=300.0, value=120.0, step=1.0)
        fhi = st.number_input("MDVP:Fhi(Hz)",min_value=100.0, max_value=600.0, value=150.0, step=1.0)
        flo = st.number_input("MDVP:Flo(Hz)",min_value=60.0, max_value=300.0, value=100.0, step=1.0)
        Jitter_percent = st.number_input("MDVP:Jitter(%)",min_value=0.0, max_value=0.03, value=0.005, step=0.001)
        Jitter_Abs = st.number_input("MDVP:Jitter(Abs)",min_value=0.0, max_value=0.002, value=0.00005, step=0.00001)
        RAP = st.number_input("MDVP:RAP",min_value=0.0, max_value=0.02, value=0.003, step=0.001)
        PPQ = st.number_input("MDVP:PPQ",min_value=0.0, max_value=0.02, value=0.003, step=0.001)
        DDP = st.number_input("Jitter:DDP",min_value=0.0, max_value=0.06, value=0.01, step=0.001)
        Shimmer = st.number_input("MDVP:Shimmer",min_value=0.0, max_value=0.2, value=0.03, step=0.005)
        Shimmer_dB = st.number_input("MDVP:Shimmer(dB)",min_value=0.0, max_value=3.0, value=0.3, step=0.1)

    with col2:
        APQ3 = st.number_input("Shimmer:APQ3",min_value=0.0, max_value=0.1, value=0.02, step=0.005)
        APQ5 = st.number_input("Shimmer:APQ5",min_value=0.0, max_value=0.15, value=0.03, step=0.005)
        APQ = st.number_input("MDVP:APQ",min_value=0.0, max_value=0.15, value=0.03, step=0.005)
        DDA = st.number_input("Shimmer:DDA",min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        NHR = st.number_input("NHR",min_value=0.0, max_value=0.5, value=0.02, step=0.01)
        HNR = st.number_input("HNR",min_value=0.0, max_value=40.0, value=20.0, step=1.0)
        RPDE = st.number_input("RPDE",min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        DFA = st.number_input("DFA",min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        spread1 = st.number_input("spread1",min_value=-10.0, max_value=0.0, value=-6.0, step=0.5)
        spread2 = st.number_input("spread2",min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        D2 = st.number_input("D2",min_value=1.0, max_value=4.0, value=2.0, step=0.1)
        PPE = st.number_input("PPE",min_value=0.0, max_value=0.5, value=0.1, step=0.01)

    model_name = st.selectbox("Select Model", list(parkiston_models.keys()))

    if st.button("Predict Parkinson Disease"):
        data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        result = predict_disease(parkiston_models[model_name], parkiston_scaler, data, parkinson_features)

        if result == 1:
            st.error("Positive: Parkinson Disease Detected")
        else:
            st.success("Negative: No Parkinson Disease")
