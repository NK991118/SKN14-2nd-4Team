
import streamlit as st
import pandas as pd
import joblib

# 저장된 scaler와 모델 로드
scaler = joblib.load('scaler.joblib')
model = joblib.load('model.joblib')

st.title("Gym 회원 이탈 예측 앱")

# 사용자 입력 UI (사이드바)
st.sidebar.header("새 회원 정보 입력")
gender = st.sidebar.selectbox("Gender", options=[0, 1])
Near_Location = st.sidebar.selectbox("Near Location", options=[0, 1])
Partner = st.sidebar.selectbox("Partner", options=[0, 1])
Promo_friends = st.sidebar.selectbox("Promo Friends", options=[0, 1])
Contract_period = st.sidebar.slider("Contract Period (months)", 1, 12, 6)
Group_visits = st.sidebar.selectbox("Group Visits", options=[0, 1])
Age = st.sidebar.slider("Age", 18, 70, 30)
Avg_additional_charges_total = st.sidebar.number_input("Avg Additional Charges Total", 0.0, 100.0, 10.0)
Month_to_end_contract = st.sidebar.slider("Month to End Contract", 0, 12, 6)
Lifetime = st.sidebar.slider("Lifetime (months)", 1, 60, 12)
Avg_class_frequency_total = st.sidebar.number_input("Avg Class Frequency Total", 0.0, 20.0, 5.0)
Avg_class_frequency_current_month = st.sidebar.number_input("Avg Class Frequency Current Month", 0.0, 20.0, 5.0)

input_df = pd.DataFrame({
    'gender': [gender],
    'Near_Location': [Near_Location],
    'Partner': [Partner],
    'Promo_friends': [Promo_friends],
    'Contract_period': [Contract_period],
    'Group_visits': [Group_visits],
    'Age': [Age],
    'Avg_additional_charges_total': [Avg_additional_charges_total],
    'Month_to_end_contract': [Month_to_end_contract],
    'Lifetime': [Lifetime],
    'Avg_class_frequency_total': [Avg_class_frequency_total],
    'Avg_class_frequency_current_month': [Avg_class_frequency_current_month]
})

if st.sidebar.button("예측하기"):
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_prob = model.predict_proba(input_scaled)[0][1]

    st.write("### 예측 결과:")
    if pred == 1:
        st.error(f"이탈 가능성 높음 (확률: {pred_prob:.2f})")
    else:
        st.success(f"이탈 가능성 낮음 (확률: {pred_prob:.2f})")
