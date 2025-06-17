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


file_path = '급식 설문조사 전체기간.csv'

df = pd.read_csv(file_path, encoding='utf-8')

# 데이터 확인
st.write(df.head())

font_url = 'https://raw.githubusercontent.com/k1ngbowser/responses/main/fonts/GmarketSansTTFLight.ttf'
font_path = 'GmarketSansTTFLight.ttf'
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

# 객관식
if '학년' in df.columns:
    st.write(df['학년'].value_counts())
if '이번주 만족도' in df.columns:
    st.write(df['이번주 만족도'].value_counts())
if '이번주 가장 좋았던 급식' in df.columns:
    st.write(df['이번주 가장 좋았던 급식'].value_counts())
if '이번주 급식이 좋았던 이유' in df.columns:
    st.write(df['이번주 급식이 좋았던 이유'].value_counts())
if '이번주 가장 싫었던 급식' in df.columns:
    st.write(df['이번주 가장 싫었던 급식'].value_counts())
if '잔반 비율' in df.columns:
    st.write(df['잔반 비율'].value_counts())
if '수면시간' in df.columns:
    st.write(df['수면시간'].value_counts())

objective_columns = ['학년', '이번주 만족도', '이번주 가장 좋았던 급식', '잔반 비율', '수면시간']

for col in objective_columns:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
        plt.title(f'{col} 응답 분포')
        plt.xlabel(col)
        plt.ylabel('응답 수')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

for col in ['이번주 만족도', '잔반 비율']:
    if col in df.columns:
        plt.figure(figsize=(6, 6))
        df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
        plt.title(f'{col} 비율')
        plt.ylabel('')
        st.pyplot(plt)

# 서술형
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

sse = []
k_range = range(2, 30)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

plt.plot(k_range, sse, 'bx-')
plt.xlabel('군집 수 (k)')
plt.ylabel('SSE (오차 제곱합)')
plt.title('엘보우 기법으로 최적 군집 수 찾기')
st.pyplot(plt)

n_clusters = 8
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
    st.write(f'\n[군집 {i} - "{cluster_keyword}"] 원본문장 목록 (총 {len(unique_originals)}건):')
    for j, row in unique_originals.iterrows():
        st.write(f'- {row["원본문장"]}')

result_df['군집명'] = result_df['군집'].map(cluster_names)
cluster_counts = result_df['군집명'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='pastel')
plt.title('서술형 응답 군집별 문장 수')
plt.xlabel('군집 키워드')
plt.ylabel('문장 수')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

reduced_df = pd.DataFrame({
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    '군집': result_df['군집'],
    '군집명': result_df['군집명']
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=reduced_df, x='x', y='y', hue='군집명', palette='tab10')
plt.title('서술형 응답 2D 군집 시각화 (PCA)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(plt)
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

file_path = '급식 설문조사 전체기간.csv'
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
objective_columns = ['학년', '이번주 만족도', '이번주 가장 좋았던 급식', '잔반 비율', '수면시간']
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
sse = []
k_range = range(2, 30)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)

elbow_df = pd.DataFrame({'k': list(k_range), 'SSE': sse})
fig = px.line(elbow_df, x='k', y='SSE', markers=True, title='엘보우 기법으로 최적 군집 수 찾기')
st.plotly_chart(fig)

# 군집화 실행
n_clusters = 8
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
    st.write(f'\n[군집 {i} - "{cluster_keyword}"] 원본문장 목록 (총 {len(unique_originals)}건):')
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
