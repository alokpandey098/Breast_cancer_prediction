import streamlit as st
import joblib
import numpy as np

# Title and description for the app
st.title('Breast Cancer Prediction System')
st.write('This app predict that person have breast cancer or not.')

# Load the pre-trained model
loaded_model = joblib.load('model.pkl')

# User input for prediction
st.write('Enter the values for prediction:')
input_data = []
# Feature names (adjust according to your model)
feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']  # Example feature names

for feature_name in feature_names:
    val = st.number_input(f"Enter {feature_name} value:", step=0.1)
    input_data.append(val)

# Perform prediction when the user clicks the button
if st.button('Predict'):
    # Reshape the input data to match the model's requirements
    input_data = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data)
    st.write(f"Predicted Output is  :> {'The tumor is Benign...' if prediction[0] == 1 else 'The tumor is Malignan... '}")
