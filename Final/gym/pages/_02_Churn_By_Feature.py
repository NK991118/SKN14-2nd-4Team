import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pickle

# 1. DNN 모델 구조 정의 (기존 PyTorch 훈련 코드와 같게)
class DNN(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(Cin, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 16)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Streamlit 앱의 전역 설정 (타이틀, 레이아웃 등)
st.set_page_config(
    layout="wide"                  # 앱 화면을 넓게 사용 (기본은 centered)
)

# 모델 및 스케일러 파일 매핑
model_scaler_files = {
    "KNeighborsClassifier": {
        "model": "models/KNeighborsClassifier.pkl",
        "scaler": "models/KNeighborsClassifier_scaler.pkl"
    },
    "LGBMClassifier": {
        "model": "models/LGBMClassifier.pkl",
        "scaler": "models/LGBMClassifier_scaler.pkl"
    },
    "LogisticRegression": {
        "model": "models/LogisticRegression.pkl",
        "scaler": "models/LogisticRegression_scaler.pkl"
    },
    "MLPClassifier": {
        "model": "models/MLPClassifier.pkl",
        "scaler": "models/MLPClassifier_scaler.pkl"
    },
    "RandomForestClassifier": {
        "model": "models/RandomForestClassifier.pkl",
        "scaler": "models/RandomForestClassifier_scaler.pkl"
    },
    "SVC": {
        "model": "models/SVC.pkl",
        "scaler": "models/SVC_scaler.pkl"
    },
    "XGBClassifier": {
        "model": "models/XGBClassifier.pkl",
        "scaler": "models/XGBClassifier_scaler.pkl"
    },
    "DNN": {
        "model": "models/best_model.pth",
        "scaler": "models/DNN_scaler.pkl"
    }

}

# 예측에 사용할 데이터셋 불러오기
df = pd.read_csv("data/test_data.csv")
df = df.drop('AgeGroup', axis=1)

# 입력 컬럼명(타겟 컬럼 'churn' 제외) 추출
input_cols = [col for col in df.columns if col != 'Churn']

# 입력 컬럼명 → 한글명 매핑 딕셔너리
column_name_map = {
    'gender': '성별',
    'Near_Location': '헬스장 근처 여부',
    'Partner': '파트너 프로그램',
    'Promo_friends': '친구 할인',
    'Contract_period': '계약 기간',
    'Group_visits': '그룹 수업 참여',
    'Age': '나이',
    'Avg_additional_charges_total': '평균 추가 요금',
    'Month_to_end_contract': '계약 만료까지 개월',
    'Lifetime': '이용 기간 개월',
    'Avg_class_frequency_total': '전체 수업 참여 빈도',
    'Avg_class_frequency_current_month': '이번 달 수업 참여 빈도'
}

# 선택형(카테고리형) 입력값(한글 → 수치) 매핑
select_options = {
    'gender': {'남': 1, '여': 0},
    'Partner': {'예': 1, '아니오': 0},
    'Near_Location': {'가까움': 1, '멀리 떨어짐': 0},
    'Promo_friends': {'추천인 있음':1, '추천인 없음':0},
    'Contract_period': {'1개월':1, '6개월':6, '12개월':12},
    'Group_visits': {'단체 방문 있음':1, '단체 방문 없음':0 },
}

float_cols = [
    'Month_to_end_contract',
    'Avg_class_frequency_total',
    'Avg_class_frequency_current_month'
]

# 전체 페이지를 15:70:15 비율로 좌-중-우 3개의 컬럼으로 나눔 (중앙 영역만 사용)
left, center, right = st.columns([1, 8, 1])

# UI 작성
with center:
    # 제목 및 안내 문구
    st.markdown("## Gym 회원 이탈 예측기")
    selected_model = st.selectbox(
        "**모델 선택**",
        list(model_scaler_files.keys())
    )
    # 입력 폼 작성
    with st.form("input_form"):
        st.markdown("### 회원 정보 입력")
        st.caption("필수 정보는 모두 입력해 주세요. 각 항목은 실제 데이터 통계를 기반으로 입력 범위가 정해집니다.")

        user_input = {}   # 사용자 입력값 저장용 딕셔너리

        # 5개씩 끊어서 container 안에 cols 5개 생성!
        for group_start in range(0, len(input_cols), 5):
            with st.container():  # 각 5개 입력창 그룹마다 컨테이너 생성 → 줄 간격/레이아웃 확보
                cols = st.columns(5)  # 한 줄에 5개의 입력 칸 생성
                # 현재 그룹에 해당하는 5개 컬럼만 반복
                for i, col in enumerate(input_cols[group_start:group_start+5]): # input_cols : 전체 컬럼 이름 저장 리스트
                    display_name = column_name_map.get(col)  # 한글 이름 가져오기, column_name_map : 한글 이름 저장된 딕셔너리
                    with cols[i]:   # 5개 컬럼 중 순서대로 배분
                        st.markdown(f"**{display_name}**")  # 입력란 제목(진하게)
                        # 선택형 입력 (성별, 파트너 여부 등)
                        if col in select_options: # col = 딕셔너리의 키 값
                            st.caption(" ")
                            options = list(select_options[col].keys())  # options : 컬럼당 밸류 값들 리스트로 저장
                            value = st.radio(
                                "",               # 라벨(컬럼)명 적는 곳
                                options,          # 라디오 버튼에 표시될 선택지, 컬럼 안의 밸류 값들이 들어감
                                index=0,          # 기본적으로 첫 번째 값 선택
                                horizontal=True,  # 가로 나열
                                key=col           # 각 입력란 고유 키값
                            )
                            user_input[col] = select_options[col][value]  # 한글→수치 변환해서 저장

                        # 수치형 입력
                        else:
                            if col in float_cols:
                                min_val = float(df[col].min())
                                max_val = float(df[col].max())
                                mean_val = float(df[col].mean())
                                st.caption(f"범위: {min_val:.2f} ~ {max_val:.2f} (평균: {mean_val:.2f})")
                            else:
                                min_val = int(df[col].min())      # 데이터셋 내 최소값
                                max_val = int(df[col].max())      # 데이터셋 내 최대값
                                mean_val = int(df[col].mean())    # 데이터셋 내 평균값(기본값)
                                st.caption(f"범위: {min_val} ~ {max_val} (평균: {mean_val})")
                            user_input[col] = st.number_input(
                                "",                   # 라벨(빈 문자열로 숨김)
                                min_value=min_val,    # 최소값 제한
                                max_value=max_val,    # 최대값 제한
                                value=mean_val,       # 초기값(평균)
                                step=0.1 if col in float_cols else 1,   # 1 or 0.1씩 증가
                                key=col               # 고유키
                            )

                        st.write("")  # 입력란 하단 여백(간격)
                        st.write("")

        # 폼 제출 버튼 및 예측 결과 처리
        submitted = st.form_submit_button("예측하기")  # 폼 제출(예측하기) 버튼
        if submitted:
            model_file = model_scaler_files[selected_model]['model']
            scaler_file = model_scaler_files[selected_model]['scaler']
            input_df = pd.DataFrame([user_input])[input_cols]

            if selected_model == "DNN":
                # DNN만 torch로 불러오기!
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)
                input_scaled_df = scaler.transform(input_df)
                Cin = input_scaled_df.shape[1]
                model = DNN(Cin)
                model.load_state_dict(torch.load(model_file, map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(input_scaled_df, dtype=torch.float32)
                    output = model(input_tensor)
                    churn_proba = torch.sigmoid(output).item()
            else:
                # 기존 sklearn 계열
                model = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
                input_scaled_df = scaler.transform(input_df)
                churn_proba = model.predict_proba(input_scaled_df)[:, 1][0]

            st.markdown("---")  # 구분선

            # === 게이지 차트(컴팩트) ===
            g_left, g_center, g_right = st.columns([2, 3, 2])
            with g_center:
                st.markdown("#### 이탈 확률(게이지)")
                fig, ax = plt.subplots(figsize=(2.1, 0.7))  # 가로, 세로 적당히 조정

                bar_color = 'crimson' if churn_proba > 0.5 else 'skyblue'

                # 0부터 확률만큼 bar 길이
                ax.barh([0], [churn_proba], color=bar_color, height=0.26, alpha=0.5)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(['0%', '50%', '100%'], fontsize=18, fontweight='bold')
                ax.set_xlabel("")
                ax.set_frame_on(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # **확률 텍스트를 바 끝에 맞게!**
                ax.text(churn_proba + 0.01, 0, f"{churn_proba:.2%}",
                        va='center', ha='left', fontsize=28, fontweight='bold', color=bar_color, alpha=0.35)

                st.pyplot(fig)

            # 예측 결과값에 따라 안내 메시지 출력
            if churn_proba > 0.5:
                st.error(f"이 회원의 이탈 확률은 **{churn_proba:.2%}** 입니다. 적극적인 케어가 필요해 보여요!")
            else:
                st.success(f"이 회원의 이탈 확률은 **{churn_proba:.2%}** 입니다. 이탈 위험이 낮은 편이에요!")