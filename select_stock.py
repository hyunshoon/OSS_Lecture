from sklearn import metrics
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, linear_model, svm)
from matplotlib.lines import Line2D # For the custom legend
from sklearn.metrics import confusion_matrix
import pandas as pd
from pykrx import stock
import model
def load_ohlcv(start, end):
    input = []
    stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')
    for i in range(start, end):
        stock = stock_list.loc[i, 'ticker']
        df = pd.read_csv(f'./data/OHLCV/{stock}.csv')
        if df.iloc[-1].touch == 1 and sum(df.iloc[-20:-1, 'touch']) == 0:
            input.append([df.loc[i, 'fluct'], df.loc[i, 'disparity20'], df.loc[i, 'upper'], df.loc[i, 'lower'],
                          df.loc[i, 'stddev']])
    return input
inputs = load_ohlcv(1000,1100)

s = '123456789'
s[-20:-1]
stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')
stock_list.iloc[-1]
for stock in stock_list:

    model.predict()