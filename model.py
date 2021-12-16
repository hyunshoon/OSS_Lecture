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

class learning:
    def __init__(self):
        self.stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')
        self.df_kospi = pd.read_csv('./data/kospi.csv')
        self.input = []
        self.label = []
        self.model = svm.SVC()
        self.today_list = []
        self.scaler = MinMaxScaler()
        # scaler = StandardScaler()
    def start_learning_make_input(self, minus):
        def is_nan(df):
            if_break = 0
            for value in df.loc[i]:#value값중 nan있으면 제외
                if type(value) == type('a'): continue
                if math.isnan(value) == True: if_break += 1
            return if_break

        for jongmok in range(0,len(self.stock_list)-minus):
            stock = self.stock_list.loc[jongmok, 'ticker']
            df = pd.read_csv(f'./data/OHLCV/{stock}.csv')
            ticker = self.stock_list.loc[jongmok].ticker#bring marketCap in stock_list
            marketCap = self.stock_list[self.stock_list['ticker'] == ticker].marketCap[jongmok]#bring marketCap in stock_list
            if len(df)<40: continue # 30거래일 이하 종목은 제외
            for i in range(20, len(df) - 10):#[start+20:date-10]
                if df.loc[i, 'touch'] == 0 or sum(df.loc[i - 20:i, 'touch']) > 1: continue  # tocuh안 한 경우 or 이전 20거래일이 touch한 적 있는 경우 s skip
                if (df.loc[i, 'disparity5'] and df.loc[i, 'disparity20'] ==100.0):continue #5이격도와 20이격도가 같은경우 결측치 처리
                if is_nan(df)>0: continue
                date = df.loc[i, '날짜']
                kospi_date = self.df_kospi[self.df_kospi.날짜 == date]
                upperWidth = (df.loc[i, 'upper'] - df.loc[i, '종가']) / df.loc[i, '종가']
                lowerWidth = (df.loc[i, 'lower'] - df.loc[i, '종가']) / df.loc[i, '종가']
                self.input.append([df.loc[i,'fluct'], df.loc[i,'disparity5'], df.loc[i,'disparity20'],df.loc[i,'disparity60'],
                              df.loc[i, 'mfi14'], upperWidth, lowerWidth,marketCap,
                              kospi_date.disparity5.values[0], kospi_date.disparity20.values[0],kospi_date.disparity60.values[0],
                              kospi_date.mfi14.values[0]])#
                if  df.loc[i + 6, '종가'] > df.loc[i+1, '시가']:
                    df.loc[i, 'label'] = 1  # 참
                    self.label.append(1)
                else:
                    df.loc[i, 'label'] = 0  # 거짓
                    self.label.append(0)
        self.input_to_df()

    def input_to_df(self):
        inputs = np.array(self.input, dtype=float)
        df = pd.DataFrame(inputs)
        self.label = np.array(self.label)
        df['label'] = self.label
        self.df = df
        self.remove_outlier()

    def remove_outlier(self):
        ori_num = len(self.df)
        for i in range(len(self.df.columns) - 1):
            q3 = np.percentile(self.df.loc[:, i], 75)
            q1 = np.percentile(self.df.loc[:, i], 25)
            IQR = (q3 - q1) * 1.5
            lowest = q1 - IQR
            highest = q3 + IQR
            outlier_idx = self.df.loc[:, i][(self.df.loc[:, i] > highest) | (self.df.loc[:, i] < lowest)].index
            self.df.drop(outlier_idx, axis=0, inplace=True)
        print(f'이상치비율: {(ori_num - len(self.df)) / ori_num * 100}%')
        self.scaling()

    def scaling(self):
        self.scaler.fit(self.df.iloc[:,0:-1])
        scaled = self.scaler.transform(self.df.iloc[:,0:-1])
        self.input = np.array(scaled, dtype=float)
        self.model_fit()

    def model_fit(self):
        self.label = np.array(self.df.loc[:, 'label'])
        x_train, x_test, y_train, y_test = train_test_split(self.input, self.label, test_size=0.25, random_state=42)

        self.model.fit(x_train, y_train)
        self.predict = self.model.predict(x_test)
        self.n_correct = sum(self.predict == y_test)
        self.accuracy = self.n_correct / len(y_test)
        print(f'정확도: {self.accuracy:.3f}%')
        print(confusion_matrix(y_test, self.predict))

    def today_stock(self, minus):
        for num in range(len(self.stock_list)-minus):
            df = pd.read_csv(f'./data/OHLCV/{self.stock_list.ticker.values[num]}.csv')
            i = df.index[-1]#마지막 행
            if df.loc[i, 'touch'] == 0: continue
            date = df.loc[i, '날짜']
            kospi_date = self.df_kospi[self.df_kospi.날짜 == date]
            upperWidth = (df.loc[i, 'upper'] - df.loc[i, '종가']) / df.loc[i, '종가']
            lowerWidth = (df.loc[i, 'lower'] - df.loc[i, '종가']) / df.loc[i, '종가']
            ticker = self.stock_list.loc[num].ticker#bring marketCap in stock_list
            marketCap = self.stock_list[self.stock_list['ticker'] == ticker].marketCap[num]#bring marketCap in stock_list
            x = [df.loc[i, 'fluct'], df.loc[i, 'disparity5'], df.loc[i, 'disparity20'], df.loc[i, 'disparity60'],
                      df.loc[i, 'mfi14'], upperWidth, lowerWidth, marketCap,
                      kospi_date.disparity5.values[0], kospi_date.disparity20.values[0], kospi_date.disparity60.values[0],
                      kospi_date.mfi14.values[0]]
            if [True for value in x if math.isnan(value)]:continue # nan 값이 있으면 제외
            x = np.array(x, dtype=float)
            x = pd.DataFrame(x)
            x['label'] = 0
            x = self.scaling_today(x)
            x = np.array(x).reshape(1, -1)
            if self.model.predict(x) == 1:#오르는 것
                self.today_list.append(stock_list.ticker.values[num])
                print(f'success:{num}')
            print(num)
        return self.today_list

    def scaling_today(self, df):
        self.scaler.fit(df.iloc[:,0:-1])
        scaled = self.scaler.transform(df.iloc[:,0:-1])
        inputs = np.array(scaled, dtype=float)
        return inputs

    # def visual(self):
    #     plt.scatter(self.inputs[:,0],self.inputs[:,1])
    #     cmap = np.array([(1, 0, 0), (0, 1, 0)])
    #     clabel = [Line2D([0], [0], marker='o', lw=0, label='상승', color=cmap[i]) for i in range(len(cmap))]
    #     #inputs : fluct, disparity5 disparity20, mfi14, upperWidth, lowerWidth, marketCap
    #     for (x, y) in [(2, 3)]:  # Not mandatory, but try [(i, i+1) for i in range(0, 30, 2)]
    #         plt.title(f'svm ({self.n_correct}/{len(self.y_test)}={self.accuracy:.3f})')
    #         plt.scatter(self.inputs[:, x], self.inputs[:, y], c=cmap[self.label], edgecolors=cmap[self.predict])
    #         plt.legend(handles=clabel, framealpha=0.5)
    #         plt.show()

if __name__ == '__main__':
    learningObj = learning()
    learningObj.start_learning_make_input(minus=2000)#minus: 시가총액 낮은순부터 제외시킬 종목 수
    today_list = learningObj.today_stock(minus=2000)
    df = stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')
    name_list = [df[df.ticker == today_list[i]].name.values[0] for i in range(len(today_list))]

