# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import font_manager
import urllib.request
import os
import plotly.express as px

file_path = '급식 설문조사 전체.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 데이터 확인
st.write(df.head())

# 폰트 설정
font_url = 'https://raw.githubusercontent.com/k1ngbowser/responses/main/fonts/GmarketSansTTFLight.ttf'
font_path = 'GmarketSansTTFLight.ttf'
if not os.path.exists(font_path):
    urllib.request.urlretrieve(font_url, font_path)
font_manager.fontManager.addfont(font_path)

# 객관식 항목 시각화
objective_columns = ['학년', '이번주 만족도', '잔반 비율', '수면시간']
for col in objective_columns:
    if col in df.columns:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, '응답 수']
        fig = px.bar(value_counts, x=col, y='응답 수', title=f'{col} 응답 분포')
        st.plotly_chart(fig)

# 파이 차트 시각화
for col in ['이번주 만족도', '잔반 비율']:
    if col in df.columns:
        pie_data = df[col].value_counts().reset_index()
        pie_data.columns = [col, '비율']
        fig = px.pie(pie_data, names=col, values='비율', title=f'{col} 비율', hole=0.4)
        st.plotly_chart(fig)

if '이번주 가장 좋았던 급식' in df.columns:
    menu_col = df['이번주 가장 좋았던 급식'].dropna().astype(str)
    week1_menus = [
    "월요일 - 마라탕, 미니육전, 초코우유, 금요일 - 참치마요덮밥, 크리스피 두부스틱,깔라만시레몬에이드", "화요일 - 순대국, 대구까스, 파인애플, 금요일 - 참치마요덮밥, 크리스피 두부스틱,깔라만시레몬에이드", "수요일 - 치킨꿔바로우, 찹쌀약과, 금요일 - 참치마요덮밥, 크리스피 두부스틱,깔라만시레몬에이드", "목요일- 찹스테이크, 산양요구르트, 금요일 - 참치마요덮밥, 크리스피 두부스틱,깔라만시레몬에이드", "금요일 - 참치마요덮밥, 크리스피 두부스틱,깔라만시레몬에이드"
]
    week2_menus = [
    "월요일 - 전주식콩나물국밥, 된장불고기, 바나나우유", "화요일 - 냉메밀국수, 알밥, 돈가스, 타코야끼, 주스", "수요일 - 육개장, 탕평채, 웅떡웅떡, 라임레몬주스, 금요일 - 부대찌개, 닭봉데리야끼구이, 요구르트(애플망고)", "목요일- 카레라이스, 왕만두, 큐브카프레제, 감자스낵", "금요일 - 부대찌개, 닭봉데리야끼구이, 요구르트(애플망고)"
]
    
    week1 = menu_col[menu_col.apply(lambda x: any(menu in x for menu in week1_menus))]
    week2 = menu_col[menu_col.apply(lambda x: any(menu in x for menu in week2_menus))]

    week_data = {
        '1주차': week1,
        '2주차': week2
    }

    for week_name, data in week_data.items():
        value_counts = data.value_counts().reset_index()
        value_counts.columns = ['급식', '응답 수']
        fig = px.bar(
            value_counts,
            x='급식', y='응답 수',
            title=f'[{week_name}] 가장 좋았던 급식 응답 분포',
            labels={'급식': '급식 메뉴'}
        )
        st.plotly_chart(fig)

# 서술형 응답 처리
text_column = '추가 메뉴와 건의사항'

def split_sentences(text):
    text = re.sub(r'[.?!]', '', text)
    return re.split(r',|그리고|또는|및|&|/|또\s+|그리고\s+', text)

split_texts = []
original_indices = []
original_sentences = []

for idx, text in df[text_column].dropna().astype(str).items():
    splits = split_sentences(text)
    for part in splits:
        cleaned = part.strip()
        if cleaned:
            split_texts.append(cleaned)
            original_indices.append(idx)
            original_sentences.append(text)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(split_texts)

# 엘보우 기법
"""sse = []
k_range = range(2, 30)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

elbow_df = pd.DataFrame({'k': list(k_range), 'SSE': sse})
fig = px.line(elbow_df, x='k', y='SSE', markers=True, title='엘보우 기법으로 최적 군집 수 찾기')
st.plotly_chart(fig)"""

# 군집화 실행
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

result_df = pd.DataFrame({
    '문장_분절': split_texts,
    '군집': labels,
    '원본문장_index': original_indices,
    '원본문장': original_sentences
})

st.write(f'\n=== [{text_column}] 군집화 결과 (군집 이름 포함) ===')
cluster_names = {}
for i in range(n_clusters):
    cluster_data = result_df[result_df['군집'] == i]
    cluster_sentences = cluster_data['문장_분절'].tolist()

    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cluster_sentences)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    top_idx = scores.argmax()
    cluster_keyword = feature_names[top_idx]

    cluster_names[i] = cluster_keyword

    unique_originals = cluster_data[['원본문장']].drop_duplicates().reset_index(drop=True)
    with st.expander(f'[군집 {i} - "{cluster_keyword}"] 응답 보기 (총 {len(unique_originals)}건):'):
    for j, row in unique_originals.iterrows():
        st.write(f'- {row["원본문장"]}')

# 군집별 문장 수 시각화
result_df['군집명'] = result_df['군집'].map(cluster_names)
cluster_counts = result_df['군집명'].value_counts().sort_values(ascending=False)
bar_df = cluster_counts.reset_index()
bar_df.columns = ['군집 키워드', '문장 수']
fig = px.bar(bar_df, x='군집 키워드', y='문장 수', title='서술형 응답 군집별 문장 수')
st.plotly_chart(fig)

# PCA 시각화
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

reduced_df = pd.DataFrame({
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    '군집': result_df['군집'],
    '군집명': result_df['군집명'],
    '문장_분절': result_df['문장_분절']
})
fig = px.scatter(
    reduced_df, x='x', y='y', color='군집명',
    title='서술형 응답 2D 군집 시각화 (PCA)',
    hover_data=['문장_분절']
)
st.plotly_chart(fig)

