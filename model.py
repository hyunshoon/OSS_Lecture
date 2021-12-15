from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, linear_model, svm)
from matplotlib.lines import Line2D # For the custom legend
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime

stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')
input = []
label = []

def make_input():
    for jongmok in range(0,len(stock_list)):
        stock = stock_list.loc[jongmok, 'ticker']
        df = pd.read_csv(f'./data/OHLCV/{stock}.csv')
        df_kospi = pd.read_csv('./data/kospi.csv')
        ticker = stock_list.loc[jongmok].ticker
        marketCap = stock_list[stock_list['ticker'] == ticker].marketCap[jongmok]#bring marketCap in stock_list
        if len(df)<40: continue # 30거래일 이하 종목은 제외
        for i in range(20, len(df) - 10):#[start+20:date-10]
            if df.loc[i, 'touch'] == 0 or sum(df.loc[i - 20:i, 'touch']) > 1: continue  # tocuh안 한 경우 or 이전 20거래일이 touch한 적 있는 경우 s skip
            if (df.loc[i, 'disparity5'] and df.loc[i, 'disparity20'] ==100.0):continue #5이격도와 20이격도가 같은경우 결측치 처리
            if_break = 0
            for value in df.loc[i]:
                if type(value) == type('a'): continue
                if math.isnan(value) == True: if_break+=1
            if if_break >0:continue
            date = df.loc[i, '날짜']
            kospi_date = df_kospi[df_kospi.날짜 == date]
            upperWidth = (df.loc[i, 'upper'] - df.loc[i, '종가']) / df.loc[i, '종가']
            lowerWidth = (df.loc[i, 'lower'] - df.loc[i, '종가']) / df.loc[i, '종가']
            # input.append([df.loc[i, 'fluct'], df.loc[i, 'disparity5'], df.loc[i, 'disparity20'],
            #               df.loc[i, 'mfi14']])
            input.append([df.loc[i,'fluct'], df.loc[i,'disparity5'], df.loc[i,'disparity20'],df.loc[i,'disparity60'],
                          df.loc[i, 'mfi14'], upperWidth, lowerWidth,marketCap,
                          kospi_date.disparity5.values[0], kospi_date.disparity20.values[0],kospi_date.disparity60.values[0],
                          kospi_date.mfi14.values[0]])#
            if  df.loc[i + 6, '종가'] > df.loc[i+1, '시가']:
                df.loc[i, 'label'] = 1  # 참
                label.append(1)
            else:
                df.loc[i, 'label'] = 0  # 거짓
                label.append(0)
make_input()
model = svm.SVC()
inputs = np.array(input, dtype=float)
label = np.array(label)
df = pd.DataFrame(inputs)
df['label'] = label

def remove_outlier(df):
    ori_num = len(df)
    for i in range(len(df.columns)-1):
        q3 = np.percentile(df.loc[:,i],75)
        q1 = np.percentile(df.loc[:,i],25)
        IQR = (q3-q1)*1.5
        lowest = q1-IQR
        highest = q3+IQR
        outlier_idx = df.loc[:,i][(df.loc[:,i] > highest) | (df.loc[:,i]< lowest)].index
        df.drop(outlier_idx, axis=0, inplace=True)
    print(f'이상치비율: {(ori_num-len(df))/ori_num*100}%')
    return df
remove_outlier(df)


def scaling(df):
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(df.iloc[:,0:-1])
    scaled = scaler.transform(df.iloc[:,0:-1])
    inputs = np.array(scaled, dtype=float)
    return inputs
inputs = scaling(df)

label = np.array(df.loc[:,'label'])
x_train, x_test, y_train, y_test = train_test_split(inputs, label, test_size=0.25, random_state=42)

model.fit(x_train, y_train)
predict = model.predict(x_test)
n_correct = sum(predict == y_test)
accuracy = n_correct / len(y_test)
print(f'정확도: {accuracy:.3f}%')
print(confusion_matrix(y_test, predict))
print(f'train_set_true_ratio: {(np.sum(y_train == 1)/(np.sum(y_train == 0)+np.sum(y_train == 1))).round(2)}%')

today_list = []
def today_stock():
    for num in range(len(stock_list)):
        df = pd.read_csv(f'./data/OHLCV/{stock_list.ticker.values[num]}.csv')
        i = df.index[-10]#마지막 행
        if df.loc[i, 'touch'] == 0: continue
        df_kospi = pd.read_csv('./data/kospi.csv')
        date = df.loc[i, '날짜']
        kospi_date = df_kospi[df_kospi.날짜 == date]
        upperWidth = (df.loc[i, 'upper'] - df.loc[i, '종가']) / df.loc[i, '종가']
        lowerWidth = (df.loc[i, 'lower'] - df.loc[i, '종가']) / df.loc[i, '종가']
        marketCap = stock_list[stock_list['ticker'] == ticker].marketCap[jongmok]#bring marketCap in stock_list
        x = [df.loc[i, 'fluct'], df.loc[i, 'disparity5'], df.loc[i, 'disparity20'], df.loc[i, 'disparity60'],
                  df.loc[i, 'mfi14'], upperWidth, lowerWidth, marketCap,
                  kospi_date.disparity5.values[0], kospi_date.disparity20.values[0], kospi_date.disparity60.values[0],
                  kospi_date.mfi14.values[0]]
        if [True for value in x if math.isnan(value)]:continue # nan 값이 있으면 제외
        x = np.array(x, dtype=float)
        x = pd.DataFrame(x)
        x['label'] = 0
        x = scaling(x)
        x = np.array(x).reshape(1, -1)
        if model.predict(x) == 1:#오르는 것
            today_list.append(stock_list.ticker.values[num])
            print(f'success:{num}')
        print(num)
today_stock()
today_list


def visual():
    plt.scatter(inputs[:,0],inputs[:,1])
    cmap = np.array([(1, 0, 0), (0, 1, 0)])
    clabel = [Line2D([0], [0], marker='o', lw=0, label='상승', color=cmap[i]) for i in range(len(cmap))]
    #inputs : fluct, disparity5 disparity20, mfi14, upperWidth, lowerWidth, marketCap
    for (x, y) in [(2, 3)]:  # Not mandatory, but try [(i, i+1) for i in range(0, 30, 2)]
        plt.title(f'svm ({n_correct}/{len(y_test)}={accuracy:.3f})')
        plt.scatter(inputs[:, x], inputs[:, y], c=cmap[label], edgecolors=cmap[predict])
        plt.legend(handles=clabel, framealpha=0.5)
        plt.show()
