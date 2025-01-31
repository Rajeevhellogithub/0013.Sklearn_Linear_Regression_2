import pickle
import numpy as np

import streamlit as st
#from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
#from streamlit.server.server import Server


# Load the Saved Regressor Model
# model = pickle.load(open(r'E:\PYTHONCLASSJUPYTER\PrakashSenapati\Salary_Prediction.pkl', 'rb'))
model = pickle.load(open(r'E:\PYTHONCLASSJUPYTER\PrakashSenapati\2024_09_23_SLR_Salary Data_Streamlit Deploy\Salary_Prediction.pkl', 'rb'))

# Set the Title of the Streamlit App
st.title('Salary Prediction App')

# Add a Brief Description
st.write('This app predicts the salary based on years of experience using a simple linear regression model')

# Add input widget for user to enter years of experience
years_experience = st.number_input('Enter Years of Experience:', min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# When the button is Clicked, Make Predictions and Display the Result
if st.button('Predict Salary'):
    # Make a prediction using the trained model
    # Convert the input to a 2D array for prediction
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)
    st.success(f'The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}')

# Display information about the model
st.write('The model was trained using a dataset of salaries and years of experience')