import pandas as pd
import numpy as np
import pickle
import streamlit as st

file1 = open('pipe.pkl','rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv('traineddata.csv')

data['IPS'].unique()

st.title('Laptop Price Predictor')

company = st.selectbox('Brand', data['Company'].unique())

type = st.selectbox('Type', data['TypeName'].unique())

ram = st.selectbox('Ram(in GB)', [2,4,6,8,12,16,24,32,64])

os = st.selectbox('OS', data['OpSys'].unique())

weight = st.number_input('Weight of the Laptop')

touchscreen = st.selectbox('Touchscreen', ['No','Yws'])

ips = st.selectbox('IPS', ['No','Yws'])

screen_size = st.number_input('Screen Size')


resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

cpu = st.selectbox('CPU', data['CPU_Name'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU(in GB)', data['Gpu Brand'].unique())

if st.button('Pridect Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips= 0
    
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = np.array([company, type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)

    
    prediction = int(np.exp(rf.predict(query)[0]))

    st.title("Predicted price for this laptop could be between " +
             str(prediction-1000)+"₹" + " to " + str(prediction+1000)+"₹")