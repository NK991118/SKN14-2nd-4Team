import streamlit as st

# 페이지 설정
st.set_page_config(page_title="회원 분석 포털", page_icon="👥", layout="wide")

# 헤더
st.markdown("""
    <h1 style="text-align: center; color: black;">👥 회원 분석 포털 👥</h1>
    <p style="text-align: center; color: black; font-size: 20px;">
        회원 현황과 이탈 예측 분석을 통해 전략적 의사결정을 지원합니다.
    </p>
""", unsafe_allow_html=True)

# 배경 이미지
image_url = "https://img.freepik.com/free-vector/data-analysis-concept-illustration_114360-5645.jpg?t=st=1745823742~exp=1745827342~hmac=fdcb5d12f06a1210a1a620d726cb3623d16527c8d4c86560f2f570d2f74c6e94&w=1060"
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="{image_url}" width="800">
    </div>
    """,
    unsafe_allow_html=True
)

# 기능 섹션
col1, col2 = st.columns(2)

# 1. 회원 현황
with col1:
    st.markdown("""
        <style>
            .user-container {
                background-color: #2196F3;  /* 파란색 배경 */
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                margin: 20px;
            }

            .user-title {
                color: white;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 10px;
            }

            .user-description {
                color: white;
                text-align: center;
                font-size: 16px;
                margin-bottom: 20px;
            }

            .user-button {
                display: inline-block;
                background-color: white;
                color: #2196F3;
                padding: 12px 30px;
                border-radius: 30px;
                text-decoration: none;
                font-weight: bold;
                font-size: 18px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: background-color 0.3s, transform 0.2s;
            }

            .user-button:hover {
                background-color: #1976D2;
                transform: scale(1.05);
            }

            .user-button:active {
                transform: scale(0.98);
            }

            a {
                text-decoration: none !important;
            }
        </style>
        <div class="user-container">
            <h3 class="user-title">📊 회원 현황</h3>
            <p class="user-description">
                전체 회원 수, 성비, 연령대 분포, 활동 지표 등 <br>
                다양한 통계를 통해 사용자 구조를 파악하세요.
            </p>
            <p style="text-align: center;">
                <a href="http://localhost:8501/01_User_Status" target="_self" class="user-button">
                    회원 현황 보기
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)

# 2. 이탈 예측
with col2:
    st.markdown("""
        <style>
            .churn-container {
                background-color: #9C27B0;  /* 보라색 배경 */
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                margin: 20px;
            }

            .churn-title {
                color: white;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 10px;
            }

            .churn-description {
                color: white;
                text-align: center;
                font-size: 16px;
                margin-bottom: 20px;
            }

            .churn-button {
                display: inline-block;
                background-color: white;
                color: #9C27B0;
                padding: 12px 30px;
                border-radius: 30px;
                text-decoration: none;
                font-weight: bold;
                font-size: 18px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: background-color 0.3s, transform 0.2s;
            }

            .churn-button:hover {
                background-color: #7B1FA2;
                transform: scale(1.05);
            }

            .churn-button:active {
                transform: scale(0.98);
            }

            a {
                text-decoration: none !important;
            }
        </style>
        <div class="churn-container">
            <h3 class="churn-title">📉 이탈 확률 조회</h3>
            <p class="churn-description">
                머신러닝 기반 분석을 통해 각 회원의 이탈 위험을 예측합니다. <br>
                타겟 마케팅과 유지 전략 수립에 활용하세요.
            </p>
            <p style="text-align: center;">
                <a href="http://localhost:8501/02_Churn_Prediction" target="_self" class="churn-button">
                    이탈 예측 보기
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)

# 푸터
st.markdown("""
    <hr style="border: 1px solid #eeeeee;">
    <p style="text-align: center; color: #888888;">문의 사항이나 피드백은 <strong>project4team@example.com</strong>으로 보내주세요.</p>
""", unsafe_allow_html=True)