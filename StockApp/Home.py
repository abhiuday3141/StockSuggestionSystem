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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler



im = Image.open('StockApp/favicon.ico')
st.set_page_config(
    page_title="Bank-Stock Prices",
    page_icon=im,
    layout="wide",
)
st.title("Stocks Prediction App")

st.write("Here I have build a web-app for 2 Models You can try both models by navigating from the Sidebar")

st.title("Stacked LSTM")
LSTM=st.expander("Know More")
with LSTM:
    st.write("A Stacked LSTM model is one where 2 or more LSTM's are connected together to improve the accuracy. The Diagram below explains it in the most simple terms")
    st.image('StockApp/StackedLSTMmodel-info.jpg')

st.title("Bi-directional LSTM")
LSTM=st.expander("Know More")
with LSTM:
    st.write("A Bi-directional LSTM is an LSTM which extracts information from both the direction For instance here it takes into account both the prices before and after for prediction and training the model. The Diagram below explains it in the most simple terms")
    st.image('StockApp/BIDirectionalLSTMmodel-info.jpg')

















