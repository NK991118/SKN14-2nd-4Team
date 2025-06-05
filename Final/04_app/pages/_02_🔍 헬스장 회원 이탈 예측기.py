import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch, pickle, os
import torch.nn as nn
from config import Config
from dnn import DNN

cfg = Config()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_abs_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)
    
st.set_page_config(
    layout="wide"                
)

df = pd.read_csv(cfg.TEST_DATA_DIR)
df = df.drop('AgeGroup', axis=1)

input_cols = [col for col in df.columns if col != 'Churn']

float_cols = [
    'Month_to_end_contract',
    'Avg_class_frequency_total',
    'Avg_class_frequency_current_month'
]
left, center, right = st.columns([2.2, 5.4, 2.2])
with center:
    st.title("🔍 헬스장 회원 이탈 예측기")

left, center, right = st.columns([1, 8, 1])
with center:

    selected_model = st.selectbox(
        "**🤖 모델 선택**",
        list(cfg.MODEL_SCALER_FILES.keys())
    )

    with st.form("input_form"):
        st.markdown("### 회원 정보 입력")
        st.caption("필수 정보는 모두 입력해 주세요. 각 항목은 실제 데이터 통계를 기반으로 입력 범위가 정해집니다.")

        user_input = {} 

        for group_start in range(0, len(input_cols), 5):
            with st.container(): 
                cols = st.columns(5)  

                for i, col in enumerate(input_cols[group_start:group_start+5]): 
                    display_name = cfg.COLUMN_NAME_MAP.get(col)  
                    with cols[i]:  
                        st.markdown(f"**{display_name}**")  

                        if col in cfg.SELECT_OPTIONS: 
                            st.caption(" ")
                            options = list(cfg.SELECT_OPTIONS[col].keys()) 
                            value = st.radio(
                                "",               
                                options,          
                                index=0,          
                                horizontal=True,  
                                key=col           
                            )
                            user_input[col] = cfg.SELECT_OPTIONS[col][value]  

                        else:
                            if col in float_cols:
                                min_val = float(df[col].min())
                                max_val = float(df[col].max())
                                mean_val = float(df[col].mean())
                                st.caption(f"범위: {min_val:.2f} ~ {max_val:.2f} (평균: {mean_val:.2f})")
                            
                            else:
                                min_val = int(df[col].min())    
                                max_val = int(df[col].max())      
                                mean_val = int(df[col].mean())   
                                st.caption(f"범위: {min_val} ~ {max_val} (평균: {mean_val})")
                            
                            user_input[col] = st.number_input(
                                "",                   
                                min_value=min_val,    
                                max_value=max_val,    
                                value=mean_val,       
                                step=0.1 if col in float_cols else 1,  
                                key=col            
                            )

                        st.write("") 
                        st.write("")

        submitted = st.form_submit_button("예측하기") 
        if submitted:
            model_file = get_abs_path(cfg.MODEL_SCALER_FILES[selected_model]['model'])
            scaler_file = get_abs_path(cfg.MODEL_SCALER_FILES[selected_model]['scaler'])
            input_df = pd.DataFrame([user_input])[input_cols]

            if selected_model == "DNN":
               
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
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)

                input_scaled_df = scaler.transform(input_df)
                churn_proba = model.predict_proba(input_scaled_df)[:, 1][0]

            st.markdown("---") 

            g_left, g_center, g_right = st.columns([2, 3, 2])
            with g_center:
                st.markdown("#### 이탈 확률(게이지)")
                fig, ax = plt.subplots(figsize=(2.1, 0.7)) 

                bar_color = 'crimson' if churn_proba > 0.5 else 'skyblue'

                ax.barh([0], [churn_proba], color=bar_color, height=0.26, alpha=0.5)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(['0%', '50%', '100%'], fontsize=18, fontweight='bold')
                ax.set_xlabel("")
                ax.set_frame_on(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                ax.text(churn_proba + 0.01, 0, f"{churn_proba:.2%}",
                        va='center', ha='left', fontsize=28, fontweight='bold', color=bar_color, alpha=0.35)

                st.pyplot(fig)

            if churn_proba > 0.5:
                st.error(f"이 회원의 이탈 확률은 **{churn_proba:.2%}** 입니다. 적극적인 케어가 필요해 보여요!")
            else:
                st.success(f"이 회원의 이탈 확률은 **{churn_proba:.2%}** 입니다. 이탈 위험이 낮은 편이에요!")