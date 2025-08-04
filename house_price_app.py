import streamlit as st
import pickle
import json
import numpy as np

# Load saved model and columns
model = pickle.load(open('house_price_model.pkl', 'rb'))
columns = json.load(open('columns.json', 'r'))['data_columns']

# Extract unique locations from columns
locations = [col.replace(' ', '').title() for col in columns if col not in ['total_sqft', 'bath', 'bhk']]

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Location column index
    if f'{location.lower()}' in columns:
        loc_index = columns.index(location.lower())
        x[loc_index] = 1

    return model.predict([x])[0]

# --- Streamlit UI ---
st.title("🏠 Bengaluru House Price Prediction App")

# Input fields
location = st.selectbox("Select Location", sorted(locations))
sqft = st.number_input("Enter Total Square Feet", min_value=500, max_value=10000, step=50)
bhk = st.slider("Select BHK", 1, 10, step=1)
bath = st.slider("Select Bathrooms", 1, 10, step=1)

if st.button("Predict Price"):
    result = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {round(result, 2)} Lakhs")
