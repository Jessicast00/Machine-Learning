import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load data dan model
df = pd.read_csv("daftar-harga-rumah-jabodetabek.csv")
with open("model_random_forest.pkl", "rb") as file:
    model = pickle.load(file)

# Konfigurasi Streamlit
st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏡 House Price Prediction - Jabodetabek")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)),url("https://images.pexels.com/photos/1486785/pexels-photo-1486785.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("Masukkan data properti untuk memprediksi harga rumah:")

land_size = st.number_input("Land (m²):", min_value=0, value=120)
build_size = st.number_input("Building (m²):", min_value=0, value=90)
bedrooms = st.slider("Bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Bathrooms:", min_value=1, max_value=10, value=2)
floors = st.slider("Floors:", min_value=1, max_value=5, value=1)
carports = st.slider("Carports:", min_value=1, max_value=5, value=1)

# City
cities = sorted(df["city"].dropna().unique())
city = st.selectbox("City:", cities)

# District, filter berdasarkan city
districts = sorted(df[df["city"] == city]["district"].dropna().unique())
district = st.selectbox("District:", districts)

# Button prediksi
if st.button("Estimate 💰"):
    features = {
        "land_size_m2": land_size,
        "building_size_m2": build_size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "carports": carports,
        "city": city,
        "district": district
    }
    log_pred = model.predict(pd.DataFrame([features]))[0]
    price_pred = np.expm1(log_pred)
    st.success(f"💡 Estimated price: Rp {price_pred:,.0f}")