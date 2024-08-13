import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model and scaler
model_file = r'C:\Mukul\Customer\Gradient Boosting_model.joblib'
scaler_file = r'C:\Mukul\Customer\scaler.joblib'

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Define a function to get user input
def get_user_input():
    st.sidebar.header('User Input Parameters')

    CreditScore = st.sidebar.slider('CreditScore', 350, 850, 650)
    Geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    Age = st.sidebar.slider('Age', 18, 100, 40)
    Tenure = st.sidebar.slider('Tenure', 0, 10, 5)
    Balance = st.sidebar.slider('Balance', 0.0, 250000.0, 100000.0)
    NumOfProducts = st.sidebar.slider('NumOfProducts', 1, 4, 2)
    HasCrCard = st.sidebar.selectbox('HasCrCard', (0, 1))
    IsActiveMember = st.sidebar.selectbox('IsActiveMember', (0, 1))
    EstimatedSalary = st.sidebar.slider('EstimatedSalary', 0.0, 200000.0, 50000.0)

    user_data = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Encode categorical variables
label_encoder = LabelEncoder()
input_df['Geography'] = label_encoder.fit_transform(input_df['Geography'])
input_df['Gender'] = label_encoder.fit_transform(input_df['Gender'])

# Standardize the features using the loaded scaler
input_df = scaler.transform(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Exited' if prediction[0] else 'Stayed')

st.subheader('Prediction Probability')
st.write(f"Stayed: {prediction_proba[0][0]:.2f}")
st.write(f"Exited: {prediction_proba[0][1]:.2f}")
