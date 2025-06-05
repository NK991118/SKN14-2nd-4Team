class Config:

    MODEL_SCALER_FILES = {
    "KNeighborsClassifier": {
        "model":  "03_trained_model/KNeighborsClassifier.pkl",
        "scaler": "03_trained_model/KNeighborsClassifier_scaler.pkl"
    },
    "LGBMClassifier": {
        "model": "03_trained_model/LGBMClassifier.pkl",
        "scaler": "03_trained_model/LGBMClassifier_scaler.pkl"
    },
    "LogisticRegression": {
        "model": "03_trained_model/LogisticRegression.pkl",
        "scaler": "03_trained_model/LogisticRegression_scaler.pkl"
    },
    "MLPClassifier": {
        "model": "03_trained_model/MLPClassifier.pkl",
        "scaler": "03_trained_model/MLPClassifier_scaler.pkl"
    },
    "RandomForestClassifier": {
        "model": "03_trained_model/RandomForestClassifier.pkl",
        "scaler": "03_trained_model/RandomForestClassifier_scaler.pkl"
    },
    "SVC": {
        "model": "03_trained_model/SVC.pkl",
        "scaler": "03_trained_model/SVC_scaler.pkl"
    },
    "XGBClassifier": {
        "model": "03_trained_model/XGBClassifier.pkl",
        "scaler": "03_trained_model/XGBClassifier_scaler.pkl"
    },
    "DNN": {
        "model": "03_trained_model/best_model.pth",
        "scaler": "03_trained_model/DNN_scaler.pkl"
    }

}
    
    COLUMN_NAME_MAP = {
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
    
    SELECT_OPTIONS = {
    'gender': {'남': 1, '여': 0},
    'Partner': {'예': 1, '아니오': 0},
    'Near_Location': {'가까움': 1, '멀리 떨어짐': 0},
    'Promo_friends': {'추천인 있음':1, '추천인 없음':0},
    'Contract_period': {'1개월':1, '6개월':6, '12개월':12},
    'Group_visits': {'단체 방문 있음':1, '단체 방문 없음':0 },
}
    ORIGINAL_DATA_DIR = '04_app/data/gym_churn_us.csv'
    TEST_DATA_DIR = '04_app/data/test_data.csv'
    KOREAN_DATA_PATH = "04_app/data/gym_test_korean.csv"
    MODEL_INPUT_PATH = "04_app/data/gym_test_for_model.csv"

