import streamlit as st

# 페이지 설정
st.set_page_config(page_title="회원 분석 포털", page_icon="👥", layout="wide")

# 상단 타이틀
st.markdown("""
    <h1 style="text-align: center; color: black;">👥 회원 분석 포털 👥</h1>
    <h1 style="text-align: center; color: black; margin-bottom: 10px; font-size: 10px;">

""", unsafe_allow_html=True)

# 상단 이미지
image_url = "https://st3.depositphotos.com/7850392/16981/v/450/depositphotos_169815306-stock-illustration-fitness-couple-man-and-woman.jpg"
st.markdown(f"""
    <div style="display: flex; justify-content: center; margin-top: -20px;">
        <img src="{image_url}" width="800">
    </div>
""", unsafe_allow_html=True)

# 스타일 공통 정의
st.markdown("""
    <style>
        .card-container {
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 20px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            color: black;
        }
        .card-container:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }

        .card-title {
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 12px;
        }

        .card-description {
            font-size: 16px;
        }

        a {
            text-decoration: none !important;
        }

        .card1 {
            background-color: #FFDEE9;
        }

        .card2 {
            background-color: #C1FFD7;
        }

        .card3 {
            background-color: #D0E6FF;
        }
    </style>
""", unsafe_allow_html=True)

# 카드 섹션
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <a href="http://localhost:8501/01_%F0%9F%93%8A_%ED%9A%8C%EC%9B%90_%ED%98%84%ED%99%A9_%EB%8C%80%EC%8B%9C%EB%B3%B4%EB%93%9C" target="_self">
            <div class="card-container card1">
                <div class="card-title">📊 회원 현황</div>
                <div class="card-description">회원 통계 및 구조 확인</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <a href="http://localhost:8501/02_%F0%9F%94%8D_%ED%97%AC%EC%8A%A4%EC%9E%A5_%ED%9A%8C%EC%9B%90_%EC%9D%B4%ED%83%88_%EC%98%88%EC%B8%A1%EA%B8%B0" target="_self">
            <div class="card-container card2">
                <div class="card-title">👟 특성별 이탈률</div>
                <div class="card-description">성별, 연령대 등 기준별 분석</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <a href="http://localhost:8501/03_%F0%9F%8F%8B%EF%B8%8F%E2%80%8D%E2%99%82%EF%B8%8F_%ED%97%AC%EC%8A%A4%EC%9E%A5_%ED%9A%8C%EC%9B%90_%EC%9D%B4%ED%83%88_%EC%98%88%EC%B8%A1" target="_self">
            <div class="card-container card3">
                <div class="card-title">🤖 모델별 예측</div>
                <div class="card-description">머신러닝 성능 비교 및 예측</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

# 하단
st.markdown("""
    <hr style="border: 1px solid #eeeeee;">
    <p style="text-align: center; color: #888888;">
        문의 사항이나 피드백은 <strong>skn_4team@example.com</strong>으로 보내주세요.
    </p>
""", unsafe_allow_html=True)
