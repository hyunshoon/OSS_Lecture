from pykrx import stock
import pandas as pd
from datetime import datetime
import time

def new_ohlcv(i):#마지막으로 저장된 날짜부터 시작
    df = pd.read_csv(f'./data/OHLCV/{stock_list.ticker.values[i]}.csv')
    if sum(df.columns == 'Unnamed: 0') != 0: df.drop(axis=0, columns='Unnamed: 0', inplace=True)
    if sum(df.columns == 'Unnamed: 0.1') != 0: df.drop(axis=0, columns='Unnamed: 0.1', inplace=True)
    if sum(df.columns == 'index') != 0: df.drop(axis=0, columns='index', inplace=True)
    now = datetime.now()
    today = str(now.year) + str(now.month) + str(now.day)
    last_date = df.날짜.values[-1].replace('-', '')
    new_df = stock.get_market_ohlcv_by_date(last_date, today, stock_list.ticker.values[i])
    new_num = df.index[-1] + 1
    new_df['날짜'] = new_df.index
    new_df['index'] = range(new_num, new_num + len(new_df))
    new_df.set_index('index', inplace=True)
    new_df['날짜'] = new_df['날짜'].apply(lambda date: str(date)[:10])  # datetime타입 str로 변경
    df = df.append(new_df)
    return plus_indicator(df, new_num)

def plus_indicator(df, new_num):
    for i in range(new_num, len(df)):
        df.loc[i, 'fluct'] = (df.loc[i, '종가'] - df.loc[i - 1, '종가']) / df.loc[i - 1, '종가'] * 100
        df.loc[i, 'MA5'] = df.loc[i-4:i,'종가'].mean()
        df.loc[i, 'MA20'] = df.loc[i-19:i,'종가'].mean()
        df.loc[i, 'MA60'] = df.loc[i-59:i,'종가'].mean()
        df.loc[i, 'disparity5'] = ((df.loc[i, '종가'] - df.loc[i, 'MA5']) / df.loc[i, 'MA5'] + 1) * 100
        df.loc[i, 'disparity20'] = ((df.loc[i, '종가'] - df.loc[i, 'MA20']) / df.loc[i, 'MA20'] + 1) * 100
        df.loc[i, 'disparity60'] = ((df.loc[i, '종가'] - df.loc[i, 'MA60']) / df.loc[i, 'MA60'] + 1) * 100
        df.loc[i, 'stddev'] = df.loc[i-19:i,'종가'].std()#20일 표준편차
        std = 2 # 볼린저밴드 표준편차 범위
        df.loc[i, 'upper'] = df.loc[i, 'MA20'] + (df.loc[i, 'stddev']*std)
        df.loc[i, 'lower'] = df.loc[i, 'MA20'] - (df.loc[i, 'stddev']*std)
        df.loc[i, 'tp'] = (df.loc[i,'시가'] + df.loc[i, '저가'] + df.loc[i, '종가']) / 3
        df.loc[i, 'pmf'] = 0
        df.loc[i, 'nmf'] = 0
        if df.loc[i-1, 'tp'] < df.loc[i, 'tp']:#plus money flow
                df.loc[i, 'pmf'] = df.loc[i, 'tp']*df.loc[i,'거래량']
        else:
                df.loc[i, 'nmf'] = df.loc[i, 'tp']*df.loc[i,'거래량']
        # MFR(money flow ratio) : 14일 동안의 pmf의 합 / 14일 동안의 nmf의 합
        df.loc[i, 'mfr'] = df.loc[i-13:i, 'pmf'].sum() / df.loc[i-13:i, 'nmf'].sum()
        df.loc[i, 'mfi14'] = 100 - (100 / (1 + df.loc[i, 'mfr']))
        df.loc[i, 'touch'] = 0
    df = df.reset_index()
    return df

def update_csv(start, end):
    for num in range(start,end):
        df = new_ohlcv(num)
        for i in range(len(df)):#볼린저밴드 벗어나는가
            if df.iloc[i].종가 >= df.iloc[i].upper or df.iloc[i].종가 <= df.iloc[i].lower:
                df.loc[i, 'touch'] = 1
        df['label'] = -100
        for i in range(20, len(df) - 10):
            if df.loc[i, 'touch'] == 0 or sum(df.loc[i - 20:i-1, 'touch']) !=0 : continue  # tocuh안 한 경우 or 이전 20거래일이 touch한 적 있는 경우 s skip
            if df.loc[i + 6, '종가'] > df.loc[i+1, '시가']:#일주일 뒤 종가가 다음날 시가보다 높으면 참
                df.loc[i, 'label'] = 1  # 참
            else:
                df.loc[i, 'label'] = 0  # 거짓
        df.to_csv(f'./data/OHLCV/{stock_list.loc[num].ticker}.csv')
        print(num)


if __name__ == '__main__':
    stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')
    stock_list.ticker.values[0]
    start = time.time()
    update_csv(0,len(stock_list))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
