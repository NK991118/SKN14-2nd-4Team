


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import pickle

# ====================== DNN 클래스 정의 ======================
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

# ====================== 파일 경로 ============================
MODEL_DIR = "../health/ml_without_phone/project/models/"
MODEL_FILES = {
    'Logistic Regression': MODEL_DIR + 'LogisticRegression.pkl',
    'Random Forest': MODEL_DIR + 'RandomForestClassifier.pkl',
    'Decision Tree': MODEL_DIR + 'DecisionTreeClassifier.pkl',
    'KNN': MODEL_DIR + 'KNeighborsClassifier.pkl',
    'SVC': MODEL_DIR + 'SVC.pkl',
    'LightGBM': MODEL_DIR + 'LGBMClassifier.pkl',
    'XGBoost': MODEL_DIR + 'XGBClassifier.pkl',
    'MLP': MODEL_DIR + 'MLPClassifier.pkl',
    'DNN': MODEL_DIR + 'best_model.pth'
}
SCALER_FILES = {
    'Logistic Regression': MODEL_DIR + 'LogisticRegression_scaler.pkl',
    'Random Forest': MODEL_DIR + 'RandomForestClassifier_scaler.pkl',
    'Decision Tree': MODEL_DIR + 'DecisionTreeClassifier_scaler.pkl',
    'KNN': MODEL_DIR + 'KNeighborsClassifier_scaler.pkl',
    'SVC': MODEL_DIR + 'SVC_scaler.pkl',
    'LightGBM': MODEL_DIR + 'LGBMClassifier_scaler.pkl',
    'XGBoost': MODEL_DIR + 'XGBClassifier_scaler.pkl',
    'MLP': MODEL_DIR + 'MLPClassifier_scaler.pkl',
    'DNN': MODEL_DIR + 'DNN_scaler.pkl'
}
KOREAN_DATA_PATH = "../health/ml_without_phone/splited_data/gym_test_korean.csv"
MODEL_INPUT_PATH = "../health/ml_without_phone/splited_data/gym_test_for_model.csv"

# ====================== 함수 정의 ============================
def load_korean_data():
    return pd.read_csv(KOREAN_DATA_PATH, encoding='utf-8-sig')

def load_model_input_data():
    return pd.read_csv(MODEL_INPUT_PATH)

def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise e

def load_scaler(path):
    try:
        return joblib.load(path)
    except Exception as e:
        # DNN 스케일러는 pickle로 저장했을 수 있으니 시도
        with open(path, 'rb') as f:
            return pickle.load(f)

def check_model_probability_support(model):
    return hasattr(model, 'predict_proba')

# ====================== Streamlit UI ========================
st.title("🏋️‍♂️ 헬스장 회원 이탈 예측")

selected_model = st.selectbox(
    "🤖 모델 선택",
    list(MODEL_FILES.keys()),
    index=0
)

try:
    korean_df = load_korean_data()
    model_input_df = load_model_input_data()

    st.info(f"📊 로드된 데이터: {korean_df.shape[0]}개 회원, {korean_df.shape[1]}개 특성")

    churn_col = '이탈 여부'
    churn_counts = korean_df[churn_col].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("현재 이용 중", churn_counts.get('유지', 0))
    with col2:
        st.metric("이탈한 회원", churn_counts.get('이탈', 0))

    with st.spinner(f"{selected_model} 모델 로딩 중..."):
        scaler = load_scaler(SCALER_FILES[selected_model])
        X_test = model_input_df.drop(columns=['Churn'])
        y_test = model_input_df['Churn']

        # DNN 분기
        if selected_model == "DNN":
            X_scaled = scaler.transform(X_test)
            Cin = X_scaled.shape[1]
            model = DNN(Cin)
            model.load_state_dict(torch.load(MODEL_FILES["DNN"], map_location="cpu"))
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).numpy().flatten()
                y_pred = (probs >= 0.5).astype(int)
                churn_probs = probs
            supports_proba = True
        else:
            model = load_model(MODEL_FILES[selected_model])
            X_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_scaled)
            supports_proba = check_model_probability_support(model)
            if supports_proba:
                churn_probs = model.predict_proba(X_scaled)[:, 1]
            else:
                churn_probs = np.full(len(X_test), np.nan)

    # ====== 성능 지표 출력 ======
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("정확도", f"{accuracy:.3f}")
    with col2:
        st.metric("정밀도", f"{precision:.3f}")
    with col3:
        st.metric("재현율", f"{recall:.3f}")
    with col4:
        st.metric("F1 점수", f"{f1:.3f}")

    # ====== 결과 표 만들기 ======
    result_df = korean_df.copy()
    result_df['예측 결과'] = ['유지' if pred == 0 else '이탈' for pred in y_pred]
    result_df['예측 정확도'] = ['정확' if actual == pred else '오류'
                           for actual, pred in zip(y_test, y_pred)]

    if supports_proba:
        result_df['이탈 확률'] = np.where(
            result_df[churn_col] == '유지',
            np.round(churn_probs, 4),
            ''
        )
    else:
        result_df['이탈 확률'] = np.where(
            result_df[churn_col] == '유지',
            'N/A (확률 미지원)',
            ''
        )
# 💰 원본 데이터 컬럼들의 계산식 설명
    with st.expander("💰 원본 데이터 컬럼 계산식"):
        st.write("""
        ### 💰 평균 추가 요금
        ```
        평균_추가_요금 = 총_추가_서비스_비용 / 총_이용_개월수
        ```
        **포함 서비스**: 개인트레이닝(PT), 락커대여, 단백질음료, 사우나, 특별프로그램, 타월서비스

        **예시**: 총 120만원 ÷ 12개월 = **월평균 10만원**

        ---

        ### 🏃‍♀️ 전체 수업 참여 빈도
        ```
        전체_수업_참여_빈도 = 총_수업_참여_횟수 / 총_이용_주수
        ```
        **포함 수업**: 요가, 필라테스, 스피닝, 에어로빅, 크로스핏, 수영강습

        **예시**: 총 65회 ÷ 24주 = **주당 2.71회**

        ---

        ### 📅 이번달 수업 참여 빈도
        ```
        이번달_수업_참여_빈도 = 이번달_수업_참여_횟수 / 4
        ```
        **측정 목적**: 최근 활동 패턴 파악, 이탈 조기 신호 감지

        **예시**: 총 6회 ÷ 4주 = **주당 1.5회**

        ### ⚠️ 이탈 위험 신호
        ```
        이번달_빈도 < 전체_빈도 × 0.5  →  활동 급감 위험
        이번달_빈도 = 0  →  활동 중단 고위험
        ```
        """)

    # 데이터 표시 옵션
    st.write("### 📋 예측 결과 상세")

    show_option = st.radio(
        "표시할 데이터 선택:",
        ["전체 데이터", "현재 이용 중인 회원만", "이탈한 회원만", "예측 실패 케이스만"]
    )

    if show_option == "전체 데이터":
        display_df = result_df
    elif show_option == "현재 이용 중인 회원만":
        display_df = result_df[result_df[churn_col] == '유지']
    elif show_option == "이탈한 회원만":
        display_df = result_df[result_df[churn_col] == '이탈']
    else:  # 예측 실패 케이스
        display_df = result_df[result_df['예측 정확도'] == '오류']

    # 컬럼 순서 재정렬 - 한글 컬럼들 먼저, 계산된 컬럼들을 지정된 순서로
    korean_feature_columns = [col for col in display_df.columns
                              if col not in ['예측 결과', '예측 정확도', '이탈 확률', churn_col]]
    calculated_columns = ['예측 결과', churn_col, '예측 정확도', '이탈 확률']
    display_columns = korean_feature_columns + calculated_columns

    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        height=400
    )

    # 고위험 회원 하이라이트 (확률 예측 지원하는 모델만)
    if show_option in ["전체 데이터", "현재 이용 중인 회원만"] and supports_proba:
        high_risk_threshold = st.slider("고위험 임계값 설정", 0.1, 0.9, 0.7, 0.1)

        # 현재 이용 중이면서 이탈 확률이 높은 회원 찾기
        high_risk_mask = (
                (result_df[churn_col] == '유지') &
                (pd.to_numeric(result_df['이탈 확률'], errors='coerce') >= high_risk_threshold)
        )
        high_risk_members = result_df[high_risk_mask]

        if len(high_risk_members) > 0:
            st.warning(f"⚠️ **고위험 회원 {len(high_risk_members)}명 발견!** (이탈 확률 >= {high_risk_threshold})")
            st.dataframe(
                high_risk_members[display_columns],
                use_container_width=True
            )
        else:
            st.success(f"✅ 이탈 확률 {high_risk_threshold} 이상인 고위험 회원이 없습니다.")
    elif not supports_proba:
        st.info(f"ℹ️ {selected_model} 모델은 확률 예측을 지원하지 않아 고위험 회원 분석을 할 수 없습니다.")

    # 모델별 추가 정보 표시
    if selected_model == 'SVC':
        st.info("""
        📝 **SVC 모델 정보:**
        - Support Vector Classifier는 기본적으로 확률 예측을 지원하지 않습니다.
        - 분류 경계를 기반으로 한 정확한 분류 결과를 제공합니다.
        - 고차원 데이터에서 우수한 성능을 보입니다.
        """)

    # 다운로드 버튼
    csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 예측 결과 CSV 다운로드",
        data=csv_data,
        file_name=f"churn_prediction_korean_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

except FileNotFoundError as e:
    st.error(f"❌ 파일을 찾을 수 없습니다: {e}")
    st.info("다음 단계를 먼저 실행해주세요:")
    st.code("""
# 1. 한글 데이터셋 생성 코드를 먼저 실행하세요
# 2. gym_test_korean.csv와 gym_test_for_model.csv 파일이 생성되었는지 확인하세요
    """)

except Exception as e:
    st.error(f"⚠️ 예상치 못한 오류가 발생했습니다: {e}")
    st.write("상세 오류 정보:")
    st.code(str(e))