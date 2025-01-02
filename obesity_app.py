import streamlit as st 
import pandas as pd 
import joblib

df = pd.read_csv(r"C:\Users\user\Desktop\Obesity Predictor App\obesity new.csv")

model_path = r"C:\Users\user\Desktop\Obesity Predictor App\ObesityPredictor.joblib"
model = joblib.load(model_path)

st.title('OBESITY DETECTOR:hospital:')
st.header('App for *early detection* of Obesity :mag_right:')

'''
**Dataset Used For The Machine Learning :page_with_curl:**
'''

st.write(df.head())

gender_map = {1: 'Male', 0: 'Female'}
df['Gender'] = df['Gender'].map(gender_map)
st.selctionbox('Gender',gender_map)

columns = ['Gender', 'Age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']
input_dict = {}

for col in columns:
    input_dict[col] = st.text_input(f'Enter value for {col}:')

if st.button('Predict'):
    prediction = model.predict([list(input_dict.values())])
    st.write(f'Predicted value: {prediction[0]}')
