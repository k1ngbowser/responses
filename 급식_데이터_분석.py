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
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import font_manager
import urllib.request
import os
import plotly.express as px

file_path = '급식 설문조사 전체-1.csv'
df = pd.read_csv(file_path, encoding='utf-8')

st.write('급식 설문조사 응답 결과/결과 분석')

# 객관식 항목 시각화
objective_columns = ['학년']
for col in objective_columns:
    if col in df.columns:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, '응답 수']
        fig = px.bar(value_counts, x=col, y='응답 수', title=f'{col} 응답 분포')
        st.plotly_chart(fig)

for col in ['이번주 만족도']:
    if col in df.columns:
        pie_data = df[col].value_counts().reset_index()
        pie_data.columns = [col, '비율']
        fig = px.pie(pie_data, names=col, values='비율', title=f'{col} 비율', hole=0.4)
        st.plotly_chart(fig)

objective_columns = ['잔반 비율', '수면시간']
for col in objective_columns:
    if col in df.columns:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, '응답 수']
        fig = px.bar(value_counts, x=col, y='응답 수', title=f'{col} 응답 분포')
        st.plotly_chart(fig)
        
# 파이 차트 시각화
for col in ['이번주 급식이 좋았던 이유','급식을 먹지 않은 이유','아침밥']:
    if col in df.columns:
        pie_data = df[col].value_counts().reset_index()
        pie_data.columns = [col, '비율']
        fig = px.pie(pie_data, names=col, values='비율', title=f'{col} 비율', hole=0.4)
        st.plotly_chart(fig)

# ---------------------- 사용자 설정 ----------------------
week1_menus = [
    "월요일 - 마라탕, 미니육전, 초코우유",
    "화요일 - 순대국, 대구까스, 파인애플",
    "수요일 - 치킨꿔바로우, 찹쌀약과",
    "목요일 - 찹스테이크, 산양요구르트",
    "금요일 - 참치마요덮밥, 크리스피 두부스틱, 깔라만시레몬에이드"
]

week2_menus = [
    "월요일 - 전주식콩나물국밥, 된장불고기, 바나나우유",
    "화요일 - 냉메밀국수, 알밥, 돈가스, 타코야끼, 주스",
    "수요일 - 육개장, 탕평채, 웅떡웅떡, 라임레몬주스",
    "목요일 - 카레라이스, 왕만두, 큐브카프레제, 감자스낵",
    "금요일 - 부대찌개, 닭봉데리야끼구이, 요구르트(애플망고)"
]

week3_menus = [
    "월요일 - 새우베이컨볶음밥, 감자샐러드, 모닝빵, 자몽에이드", 
    "화요일 - 대패삼겹버섯구이, 요구르트", 
    "수요일 - 잔치국수, 오리주물럭, 파인애플",
    "목요일 - 갈비탕, 오징어김치전, 복숭아주스",
    "금요일 - 닭갈비, 이상한나라의솜사탕아이스크림"
]

def extract_weekday_and_menu(text):
    match = re.match(r'(월요일|화요일|수요일|목요일|금요일)\s*-\s*(.+)', text.strip())
    if match:
        return match.group(1), match.group(2)
    return '기타', text.strip()

def filter_by_week(menus, target_column):
    return target_column[target_column.apply(lambda x: any(menu in x for menu in menus))]

def plot_weekday_meals(df, column_name, week_num, menus, title):
    menu_col = df[column_name].dropna().astype(str)
    week_filtered = filter_by_week(menus, menu_col)

    # 요일-식단 추출
    weekday_menu_list = week_filtered.apply(extract_weekday_and_menu)
    weekdays = [w for w, _ in weekday_menu_list]
    meals = [m for _, m in weekday_menu_list]

    df_plot = pd.DataFrame({
        '요일': weekdays,
        '식단': meals
    })

    # 요일 순서 고정
    weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일']
    df_plot['요일'] = pd.Categorical(df_plot['요일'], categories=weekday_order, ordered=True)

    # 요일별 식단 그룹별 응답 수 집계
    count_df = df_plot.groupby(['요일', '식단'], as_index=False).size()

    # 🧩 누락된 요일 채우기 위한 보완 로직
    if not count_df.empty:
        # 누락 요일-식단 조합 보완
        existing_pairs = set(zip(count_df['요일'], count_df['식단']))
        full_pairs = set((day, meal) for day in weekday_order for meal in df_plot['식단'].unique())

        missing_pairs = full_pairs - existing_pairs
        if missing_pairs:
            fill_rows = pd.DataFrame(missing_pairs, columns=['요일', '식단'])
            fill_rows['size'] = 0
            count_df = pd.concat([count_df, fill_rows], ignore_index=True)

    # 시각화
    fig = px.bar(
        count_df,
        x='요일',
        y='size',
        color='식단',
        hover_data={'식단': True, 'size': True, '요일': False},
        title=f"[{week_num}주차] {title}",
        labels={'요일': '요일', 'size': '응답 수'}
    )
    fig.update_layout(xaxis_title='요일', yaxis_title='응답 수', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Streamlit 실행 영역 ----------------------
st.title("급식 만족도-이번주 가장 좋았던 급식(2개 선택), 이번주 가장 싫었던 급식(1개 선택)")
plot_weekday_meals(df, '이번주 가장 좋았던 급식', week_num=1, menus=week1_menus, title='(1주차)이번주 가장 좋았던 급식')
plot_weekday_meals(df, '이번주 가장 좋았던 급식', week_num=2, menus=week2_menus, title='(2주차)이번주 가장 좋았던 급식')
plot_weekday_meals(df, '이번주 가장 좋았던 급식', week_num=3, menus=week3_menus, title='(3주차)이번주 가장 좋았던 급식')

plot_weekday_meals(df, '이번주 가장 싫었던 급식', week_num=1, menus=week1_menus, title='(1주차)이번주 가장 싫었던 급식')
plot_weekday_meals(df, '이번주 가장 싫었던 급식', week_num=2, menus=week2_menus, title='(2주차)이번주 가장 싫었던 급식')
 
dfl = pd.DataFrame({
    '급식을 남기는 이유': [
        '맛이없다','너무 짜거나 싱겁다',
        '메뉴선택이 진짜 최악입니다. 전 영양사분땐 안 그랬는데요.. 밥 양도 조금 주시고 반찬도 일부러 적게 주시고.. 대구까스는 진짜뭡니까..아귀강정도 뭐예요… 진짜 못먹겠어요 해산물 못먹는 친구들은 어쩌라고 주 메뉴가 죄다 해산물인가요 진짜 최악입니다',
        '양이 너무 많아서','시간이 부족하다','양이 너무 많아서','시간이 부족하다','시간이 부족하다','시간이 부족하다','너무 짜거나 싱겁다','시간이 부족하다','양이 너무 많아서','너무 짜거나 싱겁다','너무 짜거나 싱겁다','양이 너무 많아서','양이 너무 많아서',
        '시간이 부족하다','너무 짜거나 싱겁다','너무 짜거나 싱겁다','양이 너무 많아서','부실하다','양이 너무 많아서','양이 너무 많아서','시간이 부족하다','양이 너무 많아서','양이 너무 많아서','좋아하지 않는 반찬이라서','양이 너무 많아서','시간이 부족하다',
        '양이 너무 많아서','시간이 부족하다','메뉴가 별로여서','양이 너무 많아서','시간이 부족하다','양이 많은 건 아니고 그냥 배불러서 남기게 되는 것 같아용','너무 짜거나 싱겁다','너무 짜거나 싱겁다','너무 짜거나 싱겁다','양이 너무 많아서','양이 너무 많아서',
        '양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','시간이 부족하다','입에 안 맞아서','시간이 부족하다','친구들이랑 빨리 놀려고','시간이 부족하다','너무 짜거나 싱겁다',
        '간이 고등학생인데 조금만더 있었으면 좋겟어요','시간이 부족하다','너무 짜거나 싱겁다','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','맛이 없다','양이 너무 많아서','내가 안좋아 하는 음식이라서','양이 너무 많아서','거의 맨날 다 먹는다',
        '안 남겨요 맛있어요','싫어하거나 못 먹는 음식들이 있어서','너무 짜거나 싱겁다','많이 먹기엔 빨리 질림','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','너무 짜거나 싱겁다','시간이 부족하다','너무 맛있는데 누가 남겨요','싫어하는 메뉴가 있어서',
        '유제품을 못 먹음','선호하는 메뉴가 아니라서','양이 많고 먹고 싶은 메뉴가 없다 (조금만 달라고 말씀드려도 많이 주시는 경우가 있음)','양이 너무 많아서','시간이 부족하다','양이 너무 많아서','시간이 부족하다','시간이 부족하다','메뉴가 마음에 안듬','시간이 부족하다',
        '좋아하지 않는 음식이여서','양이 너무 많아서','양이 너무 많아서','시간이 부족하다','양이 너무 많아서','퀄리티가낮아서','시간이 부족하다','그냥 안먹고싶다','너무 짜거나 싱겁다','양이 너무 많아서','양이 너무 많아서','싫어하는 음식은 남김','너무 짜거나 싱겁다',
        '시간이 부족하다','먹을게 없다','맛이없다','너무 짜거나 싱겁다','시간이 부족하다','양이 너무 많아서','고기를튀긴게아니라 항상 생선이나 두부같은걸 튀기니까 맛없고 남겨요 그리고 참치마요밥 할때 깻잎 안 넣는게 좋을거같아요 친구들 반응이 호불호가 너무 많이 갈려요',
        '시간이 부족하다','시간이 부족하다','안남김','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','시간이 부족하다','국물은 다 못 먹어요','양이 너무 많아서','시간이 부족하다','맛이 없다','양이 너무 많아서','메뉴가 별로임','맛없음',
        '맛있는 건 조금 주고 맛없는 건 많이 줘서','배불러서','맛없음','시간이 부족하다','맛있는건 적게주고 맛없는것만 엄청 많이 줘서 밥이랑 같이 먹을게 없다','부실하고 맛이 애매해요ㅠ','생각보다 맛없어서','맛없어서','너무 짜거나 싱겁다','양이 너무 많아서',
        '좋아하지 않는 메뉴가 있는 경우에만','시간이 부족하다','배불러서','맛이없어서','밥은 많은데 반찬은 부족','양이 너무 많아서','너무 짜거나 싱겁다','양이 너무 많아서','맛없음','맛없음','시간이 부족하다','양이 너무 많아서','메뉴가 별로여서!','편식','너무 짜거나 싱겁다',
        '반찬은 조금만 주는데 밥은 많이 줘서(대체적으로 적게 줌. 밥을 적게 달라는게 아니라 뱐찬을 많이 달라는 뜻입니다. 고기 3점만 줄 때도 있었어요, 더 달라고하니 짜증내시고...)','너무 짜거나 싱겁다','양이 너무 많아서','양이 너무 많아서','양이 너무 많아서','너무 짜거나 싱겁다',
        '별로 안좋아하는 메뉴라서','양이 너무 많아서'
    ]
})

# 주요 3개 이유
main_reasons = ['양이 너무 많아서', '너무 짜거나 싱겁다', '시간이 부족하다']

# 카테고리 구분 함수
def categorize_reason(text):
    if text in main_reasons:
        return text
    else:
        return '기타'

dfl['카테고리'] = dfl['급식을 남기는 이유'].apply(categorize_reason)

# 카테고리별 응답 수 집계
counts = dfl['카테고리'].value_counts().reset_index()
counts.columns = ['이유', '응답 수']

# 막대그래프 시각화
fig = px.bar(
    counts,
    x='이유',
    y='응답 수',
    title='급식을 남기는 이유',
    labels={'이유': '이유', '응답 수': '응답 수'}
)

st.plotly_chart(fig, use_container_width=True)

# 기타 항목 상세 보기 (펼치기/접기)
other_texts = dfl[dfl['카테고리'] == '기타']['급식을 남기는 이유'].tolist()

if other_texts:
    with st.expander("기타 항목 펼치기 / 접기"):
        st.markdown("**기타에 포함된 응답:**")
        for i, txt in enumerate(other_texts, 1):
            st.write(f"{i}. {txt}")


# 서술형 응답 처리
def split_sentences(text):
    text = re.sub(r'[.?!]', '', text)
    return re.split(r',|그리고|또는|및|&|/|또\s+|그리고\s+', text)

split_texts = []
original_indices = []
original_sentences = []

for idx, text in df['추가 메뉴와 건의사항'].dropna().astype(str).items():
    splits = split_sentences(text)
    for part in splits:
        cleaned = part.strip()
        if cleaned:
            split_texts.append(cleaned)
            original_indices.append(idx)
            original_sentences.append(text)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = model.to('cpu')
embeddings = model.encode(split_texts)

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

st.write(f'\n=== [{'추가 메뉴와 건의사항'}] 군집화 결과 (군집 이름 포함) ===')
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

st.write('응답 결과 분석')

# 📌 Cramér's V 계산 함수
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    k = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * k))

# 📌 상관관계 분석 및 시각화 함수
def analyze_categorical_relationship(df, row_var, col_var, title):
   
    # 4. Stacked Bar Chart로 시각화
    st.markdown("### ✅ 응답별 비율 시각화 (Stacked Bar Chart)")
    proportion_df = pd.crosstab(df[row_var], df[col_var], normalize='index') * 100
    fig = px.bar(
        proportion_df,
        x=proportion_df.index,
        y=proportion_df.columns,
        barmode='stack',
        text_auto='.1f',
        labels={'value': '비율 (%)', 'index': row_var, 'variable': col_var},
        title=f"{row_var}에 따른 {col_var} 비율"
    )
    fig.update_layout(yaxis_title='비율 (%)', xaxis_title=row_var)
    st.plotly_chart(fig, use_container_width=True)
       # 1. 교차표 계산
    contingency = pd.crosstab(df[row_var], df[col_var])
    
    # 2. 카이제곱 독립성 검정
    chi2, p, dof, expected = chi2_contingency(contingency)
    v = cramers_v(contingency)

    # 3. 검정 결과 출력
    st.markdown(f"**Chi² 통계량:** {chi2:.2f}")
    st.markdown(f"**p-value:** {p:.4f}")
    st.markdown(f"**Cramér's V (관계 강도):** {v:.3f}")
    if p < 0.05:
        st.success("✔ 통계적으로 유의미한 관계입니다.")
    else:
        st.info("ℹ 통계적으로 유의미한 관계는 확인되지 않았습니다.")

def show_facet_bar(df, row_var, col_var, title):
    st.subheader(f"📊 {title}")
    fig = px.histogram(df, x=col_var, color=row_var, barmode='group', text_auto=True)
    fig.update_layout(title=title, xaxis_title=col_var, yaxis_title='응답 수')
    st.plotly_chart(fig, use_container_width=True)

       # 1. 교차표 계산
    contingency = pd.crosstab(df[row_var], df[col_var])
    
    # 2. 카이제곱 독립성 검정
    chi2, p, dof, expected = chi2_contingency(contingency)
    v = cramers_v(contingency)

    # 3. 검정 결과 출력
    st.markdown(f"**Chi² 통계량:** {chi2:.2f}")
    st.markdown(f"**p-value:** {p:.4f}")
    st.markdown(f"**Cramér's V (관계 강도):** {v:.3f}")
    if p < 0.05:
        st.success("✔ 통계적으로 유의미한 관계입니다.")
    else:
        st.info("ℹ 통계적으로 유의미한 관계는 확인되지 않았습니다.")

def show_grouped_bar(df, row_var, col_var, title):
    st.subheader(f"📊 {title}")
        
    ctab = pd.crosstab(df[row_var], df[col_var])
    ctab = ctab[sorted(ctab.columns)]  # 열 정렬

    fig = px.bar(
        ctab,
        x=ctab.index,
        y=ctab.columns,
        barmode='group',
        title=title,
        labels={'value': '응답 수', 'index': row_var},
        text_auto=True
    )
    fig.update_layout(xaxis_title=row_var, yaxis_title='응답 수')
    st.plotly_chart(fig, use_container_width=True)
       # 1. 교차표 계산
    contingency = pd.crosstab(df[row_var], df[col_var])
    
    # 2. 카이제곱 독립성 검정
    chi2, p, dof, expected = chi2_contingency(contingency)
    v = cramers_v(contingency)

    # 3. 검정 결과 출력
    st.markdown(f"**Chi² 통계량:** {chi2:.2f}")
    st.markdown(f"**p-value:** {p:.4f}")
    st.markdown(f"**Cramér's V (관계 강도):** {v:.3f}")
    if p < 0.05:
        st.success("✔ 통계적으로 유의미한 관계입니다.")
    else:
        st.info("ℹ 통계적으로 유의미한 관계는 확인되지 않았습니다.")

analyze_categorical_relationship(df, '아침밥', '이번주 만족도', '아침밥 여부와 만족도 관계')
show_facet_bar(df, '잔반 비율', '수면시간', '수면시간과 잔반 비율 관계')
analyze_categorical_relationship(df, '수면시간', '이번주 만족도', '수면시간과 만족도 관계')
show_grouped_bar(df, '잔반 비율', '이번주 만족도', '아침밥 여부와 만족도 관계')
