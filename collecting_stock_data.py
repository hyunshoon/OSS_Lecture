import pandas as pd
from pykrx import stock

date = '20211124'#이 날짜 기준으로 데이터 수집

kospi_ticker = stock.get_market_ticker_list(date, market = 'KOSPI')#2021/11/24에 상장된 코스피 종목 티커
kosdaq_ticker = stock.get_market_ticker_list(date, market = 'KOSDAQ')
tickers = kospi_ticker + kosdaq_ticker
tickers = list(map(str, tickers))#전체종목티커
names = [stock.get_market_ticker_name(ticker) for ticker in tickers]#전체종목명

dfCap = stock.get_market_cap_by_ticker(date)
marketCap = [dfCap.loc[ticker].시가총액 for ticker in tickers]#시가총액
marketCap = list(map(lambda x: x/100000000, marketCap))
marketCap = list(map(int, marketCap))

stock_df = pd.DataFrame({'id': range(len(tickers)),'ticker': tickers,'name': names, 'marketCap': marketCap})
stock_df.sort_values(by = 'marketCap', ascending=False, inplace=True)#시가총액 순으로 정렬
stock_df.id = range(len(tickers))
stock_df.set_index('id', inplace=True)
print(stock_df)
stock_df.to_csv('./data/stock_list.csv',encoding='CP949')

for ticker in stock.get_index_ticker_list(market='KOSPI'):
    print(ticker, stock.get_index_ticker_name(ticker))
for ticker in stock.get_index_ticker_list(market='KOSDAQ'):
    print(ticker, stock.get_index_ticker_name(ticker))
kospi = stock.get_index_ohlcv_by_date('20010101','20211124','1001')#코스피
kosdaq = stock.get_index_ohlcv_by_date('20010101','20211124','2001')#코스닥
kospi.to_csv('./data/kospi.csv')
kosdaq.to_csv('./data/kosdaq.csv')