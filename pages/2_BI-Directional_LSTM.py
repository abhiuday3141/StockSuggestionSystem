import math
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import tensorflow as tf
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date, timedelta
from plotly import graph_objs as go
from pandas_datareader import data,wb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import plotly.express as px
from PIL import Image
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, Embedding, Bidirectional

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Bank-Stock Prices",
    page_icon=im,
    layout="wide",
)


@st.cache
def load_data(ticker,START,TODAY):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data
def plot_raw_data(data):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock-open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock-close'))
    fig.layout.update(title_text="Time series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
def errorpercentage(y_test,pred):
    mape=np.mean(np.abs(y_test-pred)/y_test)*100
    return mape
def Make_data_set(data,timestep):
    X=[]
    Y=[]
    n=len(data)
    for i in range(0,n-timestep-1):
        X.append(data[i:i+timestep])
        Y.append(data[i+timestep])
    return np.array(X),np.array(Y)


def LSTM_Predict(bank,timestep,pred_days,type):

    START = "2010-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = load_data(bank,START,TODAY)

    df=data.reset_index()[type]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(np.array(df).reshape(-1, 1))
    train_size=int(len(df)*0.7)
    train_data=df[:train_size]
    test_data=df[train_size:]
    X_train,y_train=Make_data_set(train_data,timestep)
    X_test,y_test=Make_data_set(test_data,timestep)
    model = Sequential()
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=4,batch_size=40,verbose=1)

    predict_test=model.predict(X_test)
    error = mae(y_test,predict_test)
    predict_test=scaler.inverse_transform(predict_test)
    y_test=scaler.inverse_transform(y_test)

    idx=0
    n=len(df)
    data_for_next_prediction=df[n-timestep:]
    while idx<pred_days:
        size=len(data_for_next_prediction)
        temp=data_for_next_prediction[size-timestep:]
        temp=temp.reshape(1,timestep,1)
        val=model.predict(temp)
        data_for_next_prediction=np.append(data_for_next_prediction,val)
        idx=idx+1


    data_for_next_prediction=data_for_next_prediction.reshape(-1,1)
    data_for_next_prediction=scaler.inverse_transform(data_for_next_prediction)
    return data_for_next_prediction,predict_test,y_test,error


st.title("Bi-Directional LSTM model")
model_expander=st.expander("Know about the model used")
model_expander.write("Here we use a Bi-directional LSTM and the details about the model are as follows:")

image = Image.open('BIDirectionalLSTMmodel.jpg')
model_expander.image(image)
stocks=("Wells Fargo","JP Morgan","Citi Bank","Morgan Stanley")
Stock_mapped={
    "Wells Fargo":"WFC",
    "JP Morgan":"JPM",
    "Citi Bank":"C",
    "Morgan Stanley":"MS"
}
selected_stocks=st.selectbox("Select the Bank for prediction",stocks)
selected_stocks=Stock_mapped[selected_stocks]
START="2015-01-01"
TODAY=date.today().strftime("%Y-%m-%d")
n_days=st.slider("Days of Prediction",2,20)
timestep=st.slider("Select Timestep (the bigger the timestep the longer it would take)",3,40)
OC=("Open","Close")
type=st.selectbox("Select opening or closing price of the stock",OC)
data_load_state=st.text("Load data...")
data=load_data(selected_stocks,START,TODAY)
data_load_state.text("Loading data...done!")

def setTrue():
    st.session_state.button=True
st.subheader('Raw data')
st.write(data.head())
plot_raw_data(data)
button=st.button("Train and Predict Stock Prices",on_click=setTrue)

if 'button' not in st.session_state:
    st.session_state.button=False;
if button or st.session_state['button']:
    with st.spinner("Training..."):
        next_predicted_values,pred_test,actual_test,error=LSTM_Predict(selected_stocks,timestep,n_days,type)
        days = np.array(range(1, math.floor(n_days + 1)))
        d = 0
        dt = date.today()
        for i in range(0, n_days + 1):
            week = dt.weekday()
            if (week >= 0 and week <= 4):
                predicted_value = float(next_predicted_values[i])
                st.write("Stock value on ", dt.strftime("%d-%m-%Y"), " is", '%.2f' % predicted_value)
                i = i + 1
            dt += datetime.timedelta(1)

        grp = next_predicted_values[timestep:]
        final_graph = plt.figure(figsize=(10, 4))
        plt.plot(days, grp, marker='o')
        plt.xlabel('Days')
        plt.ylabel('Predicted Values')
        st.write(final_graph)


        buttoncomp = st.button("Show error and comparison graph")
        if buttoncomp:
            with st.spinner("Calculating Error.."):
                fig1 = plt.figure(figsize=(10, 4))
                plt.plot(actual_test)
                plt.plot(pred_test)
                plt.title('Line Graph of actual values and test values')
                ylabel = type + ' Stock Prices'
                plt.xlabel('Number of days')
                plt.ylabel(ylabel)
                plt.legend(['Orginal values', 'Test Predicted Values'], loc=2)
                st.pyplot(fig1)
                st.write("Mean absolute Error is ", error)



