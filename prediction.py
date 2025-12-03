import streamlit as st 
import pandas as pd
import numpy as np
import joblib 

# Load models and scalers
liver_models = {
    "Logistic Regression": joblib.load("liver_lr.joblib"),
    "KNN": joblib.load("liver_knn.joblib"),
    "Decision Tree": joblib.load("liver_dt.joblib"),
    "Bagging": joblib.load("liver_bg.joblib"),
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
    total_bilirubin = st.number_input("Total_Bilirubin")
    direct_bilirubin = st.number_input("Direct_Bilirubin")
    alkaline_phosphotase = st.number_input("Alkaline_Phosphotase")
    alamine_amino = st.number_input("Alamine_Aminotransferase")
    aspartate_amino = st.number_input("Aspartate_Aminotransferase")
    total_proteins = st.number_input("Total_Protiens")
    albumin = st.number_input("Albumin")
    ratio = st.number_input("Albumin_and_Globulin_Ratio")

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
            bp = st.number_input("Blood Pressure")
            sg = st.number_input("Specific Gravity")
            al = st.number_input("Albumin")
            su = st.number_input("Sugar")
            rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
            pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
            ba = st.selectbox("Bacteria", ["Present", "Not Present"])
            bgr = st.number_input("Blood Glucose Random")

    with col2:
            bu = st.number_input("Blood Urea")
            sc = st.number_input("Serum Creatinine")
            sod = st.number_input("Sodium")
            pot = st.number_input("Potassium")
            hemo = st.number_input("Hemoglobin")
            pcv = st.number_input("Packed Cell Volume")
            wc = st.number_input("White Blood Cell Count")
            rc = st.number_input("Red Blood Cell Count")
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
        fo = st.number_input("MDVP:Fo(Hz)")
        fhi = st.number_input("MDVP:Fhi(Hz)")
        flo = st.number_input("MDVP:Flo(Hz)")
        Jitter_percent = st.number_input("MDVP:Jitter(%)")
        Jitter_Abs = st.number_input("MDVP:Jitter(Abs)")
        RAP = st.number_input("MDVP:RAP")
        PPQ = st.number_input("MDVP:PPQ")
        DDP = st.number_input("Jitter:DDP")
        Shimmer = st.number_input("MDVP:Shimmer")
        Shimmer_dB = st.number_input("MDVP:Shimmer(dB)")

    with col2:
        APQ3 = st.number_input("Shimmer:APQ3")
        APQ5 = st.number_input("Shimmer:APQ5")
        APQ = st.number_input("MDVP:APQ")
        DDA = st.number_input("Shimmer:DDA")
        NHR = st.number_input("NHR")
        HNR = st.number_input("HNR")
        RPDE = st.number_input("RPDE")
        DFA = st.number_input("DFA")
        spread1 = st.number_input("spread1")
        spread2 = st.number_input("spread2")
        D2 = st.number_input("D2")
        PPE = st.number_input("PPE")

    model_name = st.selectbox("Select Model", list(parkiston_models.keys()))

    if st.button("Predict Parkinson Disease"):
        data = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        result = predict_disease(parkiston_models[model_name], parkiston_scaler, data, parkinson_features)

        if result == 1:
            st.error("Positive: Parkinson Disease Detected")
        else:
            st.success("Negative: No Parkinson Disease")
