import time, re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def today_date():
    now = datetime.now()
    return str(now.year) + str(now.month) + str(now.day)

path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf'#font_path
font_name = fm.FontProperties(fname=path, size=10).get_name()
date = today_date()#해당 날짜에 선정된 종목 선택

#선정된 종목들
today = pd.read_csv(f'{path_oss}/data/{date}_stock.csv')
name_list = list(today['name'])
code_list = list(today['code'])

name_list

def urlToList_news_dart(url):
  response = requests.get(url)
  html = response.text
  soup = BeautifulSoup(html, 'html.parser')
  li = [i.text for i in soup.find_all("a", class_ = 'tit')]
  return li

def urlTOList_influencer(url):
  response = requests.get(url)
  html = response.text
  soup = BeautifulSoup(html, 'html.parser')
  text_list = [i.text for i in soup.find_all("a", class_ = 'name_link')]#text 가져오기
  links = soup.find_all("a", class_ = 'name_link')#링크 가져오기
  link_list = []
  for link in links:
    link_list.append(link['href'])
  return zip(text_list, link_list)

def urlTOList_view(url):
  response = requests.get(url)
  html = response.text
  soup = BeautifulSoup(html, 'html.parser')
  text_list = [i.text for i in soup.find_all("a", class_ = 'api_txt_lines total_tit _cross_trigger')]#text 가져오기
  links = soup.find_all("a", class_ = 'api_txt_lines total_tit _cross_trigger')#링크 가져오기
  link_list = []
  for link in links:
    link_list.append(link['href'])
  return zip(text_list, link_list)

def crawling_to_txt():
  with open(f'{path_oss}/data/{date}stock_contents.txt', 'w')as f:
    for i in range(0,len(name_list)):
      #view,influencer, 네이버뉴스, 네이버공시정보
      urls = [(f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={name_list[i]}','view'),
            (f'https://search.naver.com/search.naver?where=influencer&sm=tab_jum&query={name_list[i]}','influencer'),
            (f'https://finance.naver.com/item/news_news.naver?code={code_list[i]}&page=&sm=title_entity_id.basic&clusterId=','news'),
            (f'https://finance.naver.com/item/news_notice.naver?code={code_list[i]}&page=','dart')]
      f.write(f'==================={name_list[i]}===================\n')
      for url in urls:
        if url[1]=='view':
          f.write("======View======\n")
          lines = urlTOList_view(url[0])
          for line in lines:
            f.write(f'{line[0]},  Link: {line[1]}\n')
        if url[1]=='influencer':
          f.write("======Influencer======\n")
          lines = urlTOList_influencer(url[0])
          for line in lines:
            f.write(f'{line[0]},  Link: {line[1]}\n')
        if url[1]=='news':
          f.write("======News======\n")
          lines = urlToList_news_dart(url[0])
          for line in lines:
            f.write(f'{line}\n')
        if url[1]=='dart':
          f.write("======Dart======\n")
          lines = urlToList_news_dart(url[0])
          for line in lines:
            f.write(f'{line}\n')
      print(f'{name_list[i]} 수집 완료')

crawling_to_txt()#크롤링 실행

"""# 종목별 키워드 수집, 정리"""

def make_keyword(name_list, code_list):
  word_list = []
  for i in range(0,len(name_list)):
    #view,influencer, 네이버뉴스, 네이버공시정보
    urls = [(f'https://search.naver.com/search.naver?where=view&sm=tab_jum&query={name_list[i]}','view'),
          (f'https://search.naver.com/search.naver?where=influencer&sm=tab_jum&query={name_list[i]}','influencer'),
          (f'https://finance.naver.com/item/news_news.naver?code={code_list[i]}&page=&sm=title_entity_id.basic&clusterId=','news'),
          (f'https://finance.naver.com/item/news_notice.naver?code={code_list[i]}&page=','dart')]
    term_list = []
    for url in urls:
      if url[1]=='view':
        lines = urlTOList_view(url[0])
      if url[1]=='influencer':
        lines = urlTOList_influencer(url[0])
      if url[1]=='news':
        lines = urlToList_news_dart(url[0])
      term_list += [okt.nouns(line[0]) for line in lines]
    term_flatten = sum(term_list, [])
    word_list.append(term_flatten)
    print(f'{name_list[i]}완료')
  return pd.DataFrame([word_list], columns=name_list)

