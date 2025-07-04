#numpang sementara
# --- Import Libraries ---
import streamlit as st 
import pandas as pd
import numpy as np 
import pickle 
import plotly.express as px 
import plotly.graph_objects as go 
from datetime import datetime, timedelta
import joblib


# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Predict Customer Satisfaction",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data ---
@st.cache_data 
def load_data():
    return pd.read_csv("data/df_no_outliers.csv")  # Perbaiki backslash jadi slash

df = load_data()
st.write("✅ File data berhasil dimuat")
st.write(df.shape)

# --- Tampilkan Data ---
st.title("Customer Satisfaction Dashboard")

st.markdown("### Contoh Data")
st.dataframe(df.head())

st.markdown("### Distribusi Satisfaction Level")
fig = px.histogram(df, x='Satisfaction_Level', color='Satisfaction_Level')
st.plotly_chart(fig, use_container_width=True)

# --- Load Models ---
@st.cache_resource
def load_all_models():
    models = {}
    models["KNN"] = joblib.load("model_baru/best_KNN_model.pkl")
    models["Decision Tree"] = joblib.load("model_baru/best_Decision_tree_model.pkl")
    models["Random Forest"] = joblib.load("model_baru/best_random_forest_model.pkl")
    return models


model_dict = load_all_models()
model_choice = st.sidebar.selectbox("Pilih Model:", list(model_dict.keys()))
selected_model = model_dict[model_choice]
st.write(f"Model yang digunakan: **{model_choice}**")

# --- Form Input User ---
st.markdown("### Input Data untuk Prediksi")

# Load encoder Country
le_country = joblib.load("model_baru/le_country_encoder.pkl")
country_labels = list(le_country.classes_)

with st.form("form_predict"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 70, 30)
        income = st.number_input("Income", min_value=0, value=50000)
        product_quality = st.selectbox("Product Quality", [1, 2, 3, 4, 5])
        service_quality = st.selectbox("Service Quality", [1, 2, 3, 4, 5])
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        purchase_frequency = st.slider("Purchase Frequency", 0, 100, 10)
        feedback_score = st.selectbox("Feedback Score", ['Low', 'Medium', 'High'])
        loyalty_level = st.selectbox("Loyalty Level", ['Bronze', 'Silver', 'Gold'])
        country = st.selectbox("Country", country_labels)  # ✅ Ganti ke label asli
    
    submitted = st.form_submit_button("Predict Satisfaction Level")

if submitted:
    # Mapping kategori ke angka sesuai label encoding
    gender_mapping = {"Male": 1, "Female": 0}
    feedback_mapping = {"Low": 0, "Medium": 1, "High": 2}
    loyalty_mapping = {"Bronze": 0, "Silver": 1, "Gold": 2}

    # Encode country menggunakan LabelEncoder
    country_encoded = le_country.transform([country])[0]

    # Buat dataframe input
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender_mapping[gender],
        "Income": income,
        "ProductQuality": product_quality,
        "ServiceQuality": service_quality,
        "PurchaseFrequency": purchase_frequency,
        "FeedbackScore": feedback_mapping[feedback_score],
        "LoyaltyLevel": loyalty_mapping[loyalty_level],
        "Country_fact": country_encoded
    }])

    # Ambil model, scaler, dan fitur
    model_obj = selected_model["model"]
    scaler = selected_model["scaler"]
    feature_order = selected_model["features"]

    # Urutkan dan scaling
    input_df = input_df[feature_order]
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model_obj.predict(input_scaled)[0]
    satisfaction_labels = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}
    pred_label = satisfaction_labels.get(prediction, prediction)

    st.success(f"✅ Prediksi Satisfaction Level: **{pred_label}**")

    # --- SHAP hanya jika Random Forest dipilih ---
    # --- SHAP hanya jika Random Forest dipilih ---
    if model_choice == "Random Forest":
        import shap
        import matplotlib.pyplot as plt
        from streamlit_shap import st_shap

        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(input_scaled)

        st.markdown("### 🔍 SHAP Force Plot")

        # Cek apakah output SHAP multi-class atau binary
        if isinstance(shap_values, list):  # Multi-class
            class_idx = prediction  # Gunakan hasil prediksi sebagai index
            base_value = explainer.expected_value[class_idx]
            shap_value = shap_values[class_idx][0]
        else:
            base_value = explainer.expected_value
            shap_value = shap_values[0]

        st_shap(shap.force_plot(
            base_value=base_value,
            shap_values=shap_value,
            features=input_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ))

        # --- SHAP Summary Plot dengan SAMPLING ---
        st.markdown("### 📊 SHAP Summary Plot (Sampled 300 rows)")

        # Sampling untuk efisiensi
        sampled_df = df[feature_order].sample(n=300, random_state=42)
        sampled_scaled = scaler.transform(sampled_df)

        # Hitung SHAP untuk sample
        with st.spinner("Menghitung SHAP summary values..."):
            shap_values_all = explainer.shap_values(sampled_scaled)

        fig_summary = plt.figure()

        if isinstance(shap_values_all, list):
            shap.summary_plot(shap_values_all[class_idx], sampled_df, show=False)
        else:
            shap.summary_plot(shap_values_all, sampled_df, show=False)

        st.pyplot(fig_summary)
    else:
        st.info("🔎 SHAP hanya tersedia untuk model Random Forest.")


    