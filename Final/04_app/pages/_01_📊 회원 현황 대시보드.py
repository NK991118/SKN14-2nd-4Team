import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib, os
from config import Config

cfg = Config()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_abs_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)

font_path = "C:/Windows/Fonts/malgun.ttf"  
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()
matplotlib.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="회원 현황", layout="wide")

left, center, right = st.columns([2.5, 4, 2.5])

with center:
    st.title("📊 회원 현황 대시보드")

data = pd.read_csv(get_abs_path(cfg.ORIGINAL_DATA_DIR))

# 0/1 컬럼 문자형 변환
data['gender_label'] = data['gender'].map({1:'남', 0:'여'})
data['Near_Location_label'] = data['Near_Location'].map({0:'No', 1:'Yes'})
data['Partner_label'] = data['Partner'].map({0:'No', 1:'Yes'})
data['Promo_friends_label'] = data['Promo_friends'].map({0:'No', 1:'Yes'})
data['Group_visits_label'] = data['Group_visits'].map({0:'No', 1:'Yes'})

# 스타일
sns.set_style("whitegrid")

# 전체 회원 수
st.subheader("👤 전체 회원 수")
st.metric("총 회원 수", f"{len(data)} 명")
st.markdown("---")

col1, col2, col3 = st.columns(3)

# 성비 시각화
with col1:
    st.subheader("🚻 성별 비율")
    gender_count = data['gender_label'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', colors=['lightblue', 'lightpink'], textprops={'fontproperties': font_prop})
    ax1.axis('equal')
    st.pyplot(fig1)

# 연령대 분포
with col2:
    st.subheader("🎂 연령대 분포")
    age_count = data['Age'].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=age_count.index, y=age_count.values, ax=ax2, palette="viridis")
    ax2.set_xlabel("나이", fontproperties=font_prop)
    ax2.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig2)

# 이용기간 분포 (Line Chart)
with col3:
    st.subheader("⏳ 이용 기간 분포")
    fig3, ax3 = plt.subplots()
    sns.histplot(data['Lifetime'], bins=30, kde=True, color='skyblue', ax=ax3)
    ax3.set_xlabel("이용 기간 (개월)", fontproperties=font_prop)
    ax3.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax3.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax3.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig3)
st.markdown("---")

col4, col5, col6 = st.columns(3)

# Near_Location (가까운 위치 여부)
with col4:
    st.subheader("📍 헬스장 근처 거주")
    fig4, ax4 = plt.subplots()
    sns.countplot(x='Near_Location_label', data=data, ax=ax4, palette="Set2")
    ax4.set_xlabel("헬스장과의 거리", fontproperties=font_prop)
    ax4.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax4.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax4.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig4)
st.markdown("---")

# Partner (제휴 여부)
with col5:
    st.subheader("🤝 제휴 회원 여부")
    fig5, ax5 = plt.subplots()
    sns.countplot(x='Partner_label', data=data, ax=ax5, palette="Set3")
    ax5.set_xlabel("제휴 여부", fontproperties=font_prop)
    ax5.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax5.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax5.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig5)

# Promo_friends (지인 추천)
with col6:
    st.subheader("👥 지인 추천 여부")
    promo_count = data['Promo_friends_label'].value_counts().sort_index()
    fig6, ax6 = plt.subplots()
    sns.barplot(x=promo_count.index, y=promo_count.values, ax=ax6, palette="Blues_d")
    ax6.set_xlabel("지인 추천 여부", fontproperties=font_prop)
    ax6.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax6.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax6.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig6)

col7, col8, col9 = st.columns(3)

# Contract_period (계약 기간)
with col7:
    st.subheader("📄 계약 기간 분포")
    fig7, ax7 = plt.subplots()
    sns.histplot(data['Contract_period'], bins=15, color="orange", ax=ax7)
    ax7.set_xlabel("계약 기간 (개월)", fontproperties=font_prop)
    ax7.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax7.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax7.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig7)
st.markdown("---")

# Group_visits (그룹 수업 참여 여부)
with col8:
    st.subheader("👨‍👩‍👧‍👦 그룹 수업 참여 여부")
    fig8, ax8 = plt.subplots()
    sns.countplot(x='Group_visits_label', data=data, ax=ax8, palette="Pastel2")
    ax8.set_xlabel("그룹 수업 참여 여부", fontproperties=font_prop)
    ax8.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax8.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax8.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig8)

# Avg_additional_charges_total (평균 추가 요금)
with col9:
    st.subheader("💰 평균 추가 요금")
    fig9, ax9 = plt.subplots()
    sns.histplot(data['Avg_additional_charges_total'], bins=30, color="coral", ax=ax9)
    ax9.set_xlabel("평균 추가 요금", fontproperties=font_prop)
    ax9.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax9.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax9.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig9)

col10, col11, col12 = st.columns(3)

# Month_to_end_contract (계약 종료까지 남은 개월 수)
with col10:
    st.subheader("📆 계약 종료까지 남은 개월 수")
    fig10, ax10 = plt.subplots()
    sns.histplot(data['Month_to_end_contract'], bins=15, color="mediumseagreen", ax=ax10)
    ax10.set_xlabel("남은 개월 수", fontproperties=font_prop)
    ax10.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax10.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax10.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig10)
st.markdown("---")

# Avg_class_frequency_total (전체 기간 수업 참석 빈도)
with col11:
    st.subheader("📚 전체 수업 참석 빈도")
    fig11, ax11 = plt.subplots()
    sns.histplot(data['Avg_class_frequency_total'], bins=30, color="slateblue", ax=ax11)
    ax11.set_xlabel("평균 수업 참석 횟수", fontproperties=font_prop)
    ax11.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax11.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax11.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig11)

# Avg_class_frequency_current_month (이번 달 수업 참석 빈도)
with col12:
    st.subheader("📅 이번 달 수업 참석 빈도")
    fig12, ax12 = plt.subplots()
    sns.histplot(data['Avg_class_frequency_current_month'], bins=30, color="lightgreen", ax=ax12)
    ax12.set_xlabel("이번 달 수업 횟수", fontproperties=font_prop)
    ax12.set_ylabel("회원 수", fontproperties=font_prop)
    for label in ax12.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax12.get_yticklabels():
        label.set_fontproperties(font_prop)
    st.pyplot(fig12)
