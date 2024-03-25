import streamlit as st
import numpy as np
import joblib
#from sklearn.metrics._dist_metrics import EuclideanDistance


scaler = joblib.load('Scaler.joblib')
#knn = joblib.load('KNN.pkl')
#rf = joblib.load('RF.pkl')
logistic = joblib.load('Logistic.joblib')
svc = joblib.load('SVC.joblib')

st.title('Heart Disease Prediction Model')

sex_options = ['M', 'F']
chest_pain_options = ['ATA', 'NAP', 'ASY', 'TA']
resting_ecg_options = ['Normal', 'ST', 'LVH']
exercise_angina_options = ['N','Y']
st_slope_options = ['Up', 'Flat', 'Down']

age = st.number_input('Age')
sex = st.selectbox('Sex', sex_options)
chest_pain_type = st.selectbox('Chest Pain Type', chest_pain_options)
resting_bp = st.number_input('Resting Blood Pressure')
cholesterol = st.number_input('Cholesterol')
fasting_bs = st.number_input('Fasting Blood Sugar')
resting_ecg = st.selectbox('Resting ECG', resting_ecg_options)
max_hr = st.number_input('Max Heart Rate')
exercise_angina = st.selectbox('Exercise Induced Angina', exercise_angina_options)
oldpeak = st.number_input('Oldpeak')
st_slope = st.selectbox('ST Slope', st_slope_options)

if st.button('Predict'):
    input = [age,sex,chest_pain_type,resting_bp,cholesterol,fasting_bs,resting_ecg,max_hr,exercise_angina,oldpeak,st_slope]
    sex = {'M':0,'F':1}
    chest = {'ATA':0, 'NAP':1, 'ASY':2, 'TA':3}
    ecg = {'Normal':0, 'ST':1, 'LVH':2}
    exercise = {'N':0, 'Y':1}
    slope = {'Up':0, 'Flat':1, 'Down':2}
    input[1] = sex.get(input[1],input[1])
    input[2] = chest.get(input[2],input[2])
    input[6] = ecg.get(input[6],input[6])
    input[8] = exercise.get(input[8],input[8])
    input[10] = slope.get(input[10],input[10])

    input = [input]
    input = np.array(input)
    input = scaler.transform(input)

    res = logistic.predict(input)
    st.write('Result of KNN : ' + str(res[0]))
    st.write('Result of Random Forest : ' + str(res[0]))
    st.write('Result of Logistic Regression : ' + str(res[0]))
    
    res = svc.predict(input)
    st.write('Result of Support Vector Machine : ' + str(res[0]))
    st.write('0 means safe from heart diseases')
    st.write('1 means susceptable to heart diseases')

