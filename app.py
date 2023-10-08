import streamlit as st
import pickle
import numpy as np

# importing the model
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))


#website designing
st.title("Laptop Price Predictor")
#creating dropdown boxex for different specifications
Company = st.selectbox('Company',df['Company'].unique())
Type = st.selectbox('Type',df['TypeName'].unique())
Ram = st.selectbox('Ram',[2,4,6,8,12,16,24,32,64])
Weight = st.number_input('Weight of the Laptop')
TouchScreen = st.selectbox('TouchScreen',['Yes','No'])
Ips = st.selectbox('IPS',['Yes','No'])
ScreenSize = st.number_input('Screen Size')
Resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900',
                                               '3840x2160','3200x1800','2880x1800','2560x1600',
                                               '2560x1440','2304x1440'])
CpuBrand = st.selectbox('CPU Brand',df['Cpu brand'].unique())
HDD = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
SSD = st.selectbox('SSD(in GB)',[0,128,256,512,1024])
GPU = st.selectbox('GPU',df['Gpu brand'].unique())
OS = st.selectbox('OS',df['os'].unique())

#Button for submission of details
if st.button('Predict Price'):
    # query
    Ppi = None
    if TouchScreen == 'Yes':
        TouchScreen = 1
    else:
        TouchScreen = 0

    if Ips == 'Yes':
        Ips = 1
    else:
        Ips = 0

    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    Ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / ScreenSize

    query = np.array([Company, Type, Ram, Weight, TouchScreen, Ips, Ppi, CpuBrand, HDD, SSD, GPU, OS])

    query = query.reshape(1, 12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))











