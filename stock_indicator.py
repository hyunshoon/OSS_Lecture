import numpy as np
import pandas as pd
from pykrx import stock

stock_list = pd.read_csv('./data/stock_list.csv', encoding='CP949')

def making_indicator(df):
    fluct = (df['종가'] - df['종가'].shift(1)) / df['종가'] * 100  # 일별 등락률 추가
    df['fluct'] = fluct  # 당일 등락률
    df['MA5'] = df['종가'].rolling(window=5, min_periods=1).mean()  # 5일 이동평균선
    df['MA20'] = df['종가'].rolling(window=20, min_periods=1).mean()  # 이동평균선
    df['MA60'] = df['종가'].rolling(window=60, min_periods=1).mean()  # 이동평균선
    df['disparity5'] = ((df['종가'] - df['MA5']) / df['MA5'] + 1) * 100  # 이격도
    df['disparity20'] = ((df['종가'] - df['MA20']) / df['MA20'] + 1) * 100  # 이격도
    df['disparity60'] = ((df['종가'] - df['MA60']) / df['MA60'] + 1) * 100  # 이격도
    df['stddev'] = df['종가'].rolling(window=20, min_periods=1).std()  # 표준편차
    df['upper'] = df['MA20'] + (df['stddev'] * 2)
    df['lower'] = df['MA20'] - (df['stddev'] * 2)
    df['tp'] = (df['시가'] + df['저가'] + df['종가']) / 3
    df['pmf'] = 0  # i 번째 tp < i+1 번째 tp 인경우 p(positive)mf -> 긍정적인 현금흐름
    df['nmf'] = 0  # i 번째 tp >= i+1 번째 tp 인경우 n(negative)mf -> 부정적인 현금흐름
    for i in range(len(df['종가']) - 1):
        if df.tp.values[i] < df.tp.values[i + 1]:
            # 중심가*거래량
            df.pmf.values[i + 1] = df.tp.values[i + 1] * df.거래량.values[i + 1]
            df.nmf.values[i + 1] = 0
        else:
            # df.tp.values[i] >= df.tp.values[i + 1]
            df.nmf.values[i + 1] = df.tp.values[i + 1] * df.거래량.values[i + 1]
            df.pmf.values[i + 1] = 0
    # MFR(money flow ratio) : 14일 동안의 pmf의 합 / 14일 동안의 nmf의 합
    df['mfr'] = df.pmf.rolling(window=14).sum() / df.nmf.rolling(window=14).sum()
    # MFI14 : 14일 동안의 현금 흐름 100 - (100 / ( 1 + mfr ))
    df['mfi14'] = 100 - (100 / (1 + df.mfr))
    df['touch'] = 0
    df = df.reset_index()
    return df

def pykrx_to_df(num):
    ticker = stock_list.loc[num].ticker
    df = stock.get_market_ohlcv_by_date(start, date, ticker)
    df = making_indicator(df)
    return df
def data_to_csv(start, end):
    for num in range(start,end):
        df = pykrx_to_df(num)
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

#1500까지함
if __name__ == '__main__':
    date = '20211217'  # 이 날짜 기준으로 데이터 수집
    start = '20180101'
    # data_to_csv(1500,len(stock_list))
    #making index
