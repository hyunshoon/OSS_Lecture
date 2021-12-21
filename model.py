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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import time
import matplotlib.font_manager as fm


stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')

'''
이상치비율: 10.379128202596634%
train_set:27231개, test_set:9078개
정확도: 57.920%
[[4273  606]
 [3214  985]]
time : 519.2407331466675
'''



def today_date():
    now = datetime.now()
    return str(now.year) + str(now.month) + str(now.day)


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
        def pass_today():
            if_break = 0
            if self.ohlcv.loc[i, 'touch'] == 0 or sum(self.ohlcv.loc[i - 20:i, 'touch']) > 1: if_break += 1 # tocuh안 한 경우 or 이전 20거래일이 touch한 적 있는 경우 s skip
            if (self.ohlcv.loc[i, 'disparity5'] and self.ohlcv.loc[i, 'disparity20'] == 100.0): if_break += 1  # 5이격도와 20이격도가 같은경우 결측치 처리
            for value in self.ohlcv.loc[i]:#value값중 nan있으면 제외
                if type(value) == type('a'): continue
                if math.isnan(value) == True: if_break += 1
            return if_break
        def labeling():
            if self.ohlcv.loc[i + 5, '종가'] > self.ohlcv.loc[i, '종가']:
                # self.ohlcv.loc[i, 'label'] = 0  # 참
                self.label.append(0)#참
            else:
                # self.ohlcv.loc[i, 'label'] = 1  # 거짓
                self.label.append(1)#거짓
        for jongmok in range(0,len(self.stock_list)-minus):
            stock = self.stock_list.loc[jongmok, 'ticker']
            self.ohlcv = pd.read_csv(f'./data/OHLCV/{stock}.csv')
            ticker = self.stock_list.loc[jongmok].ticker#bring marketCap in stock_list
            marketCap = self.stock_list[self.stock_list['ticker'] == ticker].marketCap[jongmok]#bring marketCap in stock_list
            if len(self.ohlcv)<40: continue # 30거래일 이하 종목은 제외
            for i in range(20, len(self.ohlcv) - 10):#[start+20:date-10]
                if pass_today()>0: continue
                date = self.ohlcv.loc[i, '날짜']
                kospi_date = self.df_kospi[self.df_kospi.날짜 == date]
                upperWidth = (self.ohlcv.loc[i, 'upper'] - self.ohlcv.loc[i, '종가']) / self.ohlcv.loc[i, '종가']
                lowerWidth = (self.ohlcv.loc[i, 'lower'] - self.ohlcv.loc[i, '종가']) / self.ohlcv.loc[i, '종가']
                self.input.append([self.ohlcv.loc[i,'fluct'], self.ohlcv.loc[i,'disparity5'], self.ohlcv.loc[i,'disparity20'],self.ohlcv.loc[i,'disparity60'],
                              self.ohlcv.loc[i, 'mfi14'], upperWidth, lowerWidth,marketCap,
                              kospi_date.disparity5.values[0], kospi_date.disparity20.values[0],kospi_date.disparity60.values[0],
                              kospi_date.mfi14.values[0]])#
                labeling()
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
            IQR = (q3 - q1) * 1.5 *2 #1.5가 표준이지만 극단값을 제거하기 위함이므로 더 높게 설정한다.
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
        self.accuracy = self.n_correct / len(y_test) *100
        print(f'train_set:{len(x_train)}개, test_set:{len(x_test)}개')
        print(f'정확도: {self.accuracy:.3f}%')
        print(confusion_matrix(y_test, self.predict))#실제값, 예측값

        #confusion matrix visualization
        # label = ['Down', 'Up']  # 라벨 설정
        label = ['Up', 'Down']  # 라벨 설정
        plot = plot_confusion_matrix(self.model,  # 분류 모델
                                     x_test,y_test,   # 예측 데이터와 예측값의 정답(y_true)
                                     display_labels=label,  # 표에 표시할 labels
                                     # cmap=plt.cm.Blue,  # 컬러맵(plt.cm.Reds, plt.cm.rainbow 등이 있음)
                                     normalize=None)  # 'true', 'pred', 'all' 중에서 지정 가능. default=None
        plot.ax_.set_title('Confusion Matrix')
        plt.show()

    def today_stock(self, minus):
        for num in range(len(self.stock_list)-minus):
            df = pd.read_csv(f'./data/OHLCV/{self.stock_list.ticker.values[num]}.csv')
            i = df.index[-1]#마지막 행
            if df.loc[i, 'touch'] == 0 or sum(df.loc[i - 20:i, 'touch']) > 1: continue #밴드를 최근 20거래일중 오늘 최초로 터치했을 때
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
            x = pd.DataFrame(np.array(x, dtype=float))
            x['label'] = 0 #dataframe format 맞춰주는 용도
            x = self.scaling_today(x)
            x = np.array(x).reshape(1, -1)
            today_down = []
            if self.model.predict(x) == 0:#오르는 것
                self.today_list.append(stock_list.ticker.values[num])
                print(f'success:{num}')
            else:#에러 확인용.
                today_down.append(stock_list.ticker.values[num])
        return self.today_list, today_down

    def scaling_today(self, df):
        self.scaler.fit(df.iloc[:, 0:-1])
        scaled = self.scaler.transform(df.iloc[:, 0:-1])
        # self.scaler.fit(df)
        # scaled = self.scaler.transform(df)
        inputs = np.array(scaled, dtype=float)
        return inputs


if __name__ == '__main__':
    start = time.time()
    learningObj = learning()
    minus = 0#minus: 시가총액 낮은순부터 제외시킬 종목 수
    learningObj.start_learning_make_input(minus)
    code_list, down_list = learningObj.today_stock(minus)
    df =pd.read_csv('./data/stock_list.csv', encoding='CP949')
    name_list = [df[df.ticker == code_list[i]].name.values[0] for i in range(len(code_list))]#ticker로 종목명 불러오기
    today_df = pd.DataFrame(zip(name_list, code_list), columns=['name','code'])
    today_df.to_csv(f'./data/selected_stock/{today_date()}_stock.csv', encoding='utf-8-sig')
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

