import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ABC Retail Analytics", page_icon="🛍️", layout="centered")

@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load('rf_model.pkl'),
        "kmeans": joblib.load('kmeans_model.pkl'),
        "scaler": joblib.load('scaler.pkl'),
        "encoders": joblib.load('label_encoders.pkl'),
        "features": joblib.load('feature_names.pkl')
    }

try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please execute 'train_model.py' inside your environment terminal first.")
    st.stop()

st.title("🛍️ Black Friday Purchase Predictor")
st.subheader("ABC Private Limited — Customer Intelligence Engine")
st.write("Input demographics and target product parameters below to output an optimized purchase amount value.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.selectbox("Age Bracket", ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"])
    occupation = st.slider("Occupation Code ID", 0, 20, 5)
    city_cat = st.selectbox("City Category classification", ["A", "B", "C"])

with col2:
    stay_years = st.selectbox("Duration of Stay in Current City (Years)", ["0", "1", "2", "3", "4+"])
    marital = st.selectbox("Marital Status", ["Unmarried", "Married"])
    prod_cat_1 = st.slider("Primary Product Category (1-20)", 1, 20, 1)
    prod_cat_2 = st.slider("Secondary Product Category (Optional, 0 if None)", 0, 20, 0)
    prod_cat_3 = st.slider("Tertiary Product Category (Optional, 0 if None)", 0, 20, 0)

if st.button("Generate Optimization Metrics"):
    # Encode inputs
    gender_enc = artifacts["encoders"]["Gender"].transform([gender])[0]
    age_enc = artifacts["encoders"]["Age"].transform([age])[0]
    city_enc = artifacts["encoders"]["City_Category"].transform([city_cat])[0]
    stay_numeric = int(stay_years.replace('+', ''))
    marital_numeric = 1 if marital == "Married" else 0
    
    # Assembly vectors
    demographic_vector = np.array([[gender_enc, age_enc, occupation, city_enc, stay_numeric, marital_numeric]])
    
    # Formulate cluster group assignment dynamically
    scaled_demo = artifacts["scaler"].transform(demographic_vector)
    assigned_cluster = artifacts["kmeans"].transform(scaled_demo).argmin(axis=1)[0]
    
    # Final target predictive inference payload assembly
    full_input_payload = pd.DataFrame([{
        'Gender': gender_enc, 'Age': age_enc, 'Occupation': occupation,
        'City_Category': city_enc, 'Stay_In_Current_City_Years': stay_numeric,
        'Marital_Status': marital_numeric, 'Product_Category_1': prod_cat_1,
        'Product_Category_2': prod_cat_2, 'Product_Category_3': prod_cat_3,
        'User_Cluster': assigned_cluster
    }])
    
    # Generate prediction safely mapping correct order structure array formats
    final_features_ordered = full_input_payload[artifacts["features"]]
    predicted_spend = artifacts["model"].predict(final_features_ordered)[0]
    
    # Cluster UI display definitions
    cluster_names = {0: "Budget-Conscious Browser", 1: "Mainstream Consistent Consumer", 2: "High-Value Power Shopper"}
    
    st.markdown("---")
    st.success(f"### 🎯 Predicted Purchase Amount: ₹{predicted_spend:,.2f}")
    
    st.info(f"💡 **Assigned Persona:** {cluster_names.get(assigned_cluster, 'Standard Profile')}")
    st.caption("Use this output value threshold directly to trigger targeted high-conversion promo discounts.")