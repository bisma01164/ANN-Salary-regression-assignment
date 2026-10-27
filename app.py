import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load all
model = load_model("model.h5",compile=False)
with open("StandardScaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("LabelEncoder_Gender.pkl.pkl", "rb") as f:
    le_gender = pickle.load(f)
with open("onehot_encoder_geo.pkl", "rb") as f:
    ohe_geo = pickle.load(f)

st.title("Salary Prediction")

# Inputs
geography = st.selectbox("Geography", ohe_geo.categories_[0])
gender = st.selectbox("Gender", le_gender.classes_)
credit = st.number_input("CreditScore", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict Salary"):
    # Preprocess same as training
    gender_encoded = le_gender.transform([gender])[0]
    geo_encoded = ohe_geo.transform([[geography]]).toarray() if hasattr(ohe_geo.transform([[geography]]), 'toarray') else ohe_geo.transform([[geography]])
    # For OneHotEncoder that returns array directly
    try:
        geo_encoded = ohe_geo.transform([[geography]])
        if hasattr(geo_encoded, 'toarray'):
            geo_encoded = geo_encoded.toarray()
    except:
        geo_encoded = ohe_geo.transform([[geography]])

    # Data in EXACT same order as training: CreditScore,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,Geo
    input_data = pd.DataFrame([[
        credit, gender_encoded, age, tenure, balance, num_products, has_card, is_active
    ]], columns=['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember'])
    
    geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
    final_input = pd.concat([input_data, geo_df], axis=1)
    
    # Debug - terminal me dekhega
    print(f"Final input shape: {final_input.shape}, columns: {final_input.columns.tolist()}")
    print(f"Scaler expects: {scaler.n_features_in_}, Model expects: {model.input_shape}")

    scaled = scaler.transform(final_input)
    pred = model.predict(scaled, verbose=0)
    st.success(f"Estimated Salary: ${pred[0][0]:,.2f}")