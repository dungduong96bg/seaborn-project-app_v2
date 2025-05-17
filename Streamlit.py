import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, date
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load mô hình LightGBM
#model = lgb.Booster(model_file=r'C:\Users\ADMIN\Documents\final_model.txt')
model = lgb.Booster(model_file='final_model.txt')

# ---------------------- Hàm xử lý ----------------------
def encode_education(level):
    mapping = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3,
        "Academic degree": 4
    }
    return mapping.get(level, np.nan)

def encode_gender(gender):
    return 0 if gender == "F" else 1 if gender == "M" else np.nan

def encode_days_employed(days):
    return days if days != 365243 else np.nan

def compute_debt_credit_ratio(debt, credit_sum):
    return debt / credit_sum if credit_sum != 0 else 0

def compute_days_birth(birth_date):
    return (birth_date - datetime.today().date()).days

def compute_term(credit_amount, annuity):
    if annuity and annuity != 0:
        return annuity, round(credit_amount / annuity)
    return None, None

def calculate_days_id_publish(change_date_str, application_date_str):
    change_date = datetime.strptime(change_date_str, '%Y-%m-%d')
    application_date = datetime.strptime(application_date_str, '%Y-%m-%d')
    return (application_date - change_date).days * -1

def calculate_days_credit(application_date):
    return (application_date - datetime.today().date()).days

def calculate_days_credit_enddate_max(credit_durations):
    return max(credit_durations) * -1

def map_income_type(income_type):
    mapping = {
        'Working': '0', 'Unemployed': '1', 'Student': '2',
        'State servant': '3', 'Pensioner': '4', 'Maternity leave': '5',
        'Commercial associate': '6', 'Businessman': '7'
    }
    return float(mapping.get(income_type, np.nan))

def convert_document_3(value):
    return 1 if value == 'Có' else 0

def convert_car_ownership(value):
    return 1 if value == 'Có' else 0

def suggest_credit_limit(score_100, max_amount=500_000_000):
    if score_100 >= 80:
        return 0.8 * max_amount
    elif score_100 >= 60:
        return 0.6 * max_amount
    elif score_100 >= 40:
        return 0.4 * max_amount
    else:
        return 0.2 * max_amount

# ---------------------- UI ----------------------
st.set_page_config(page_title="Credit Scoring App", layout="centered")
st.title("💳 Ứng dụng chấm điểm tín dụng khách hàng")
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;  /* Đen tuyệt đối */
        color: #cccccc;  /* Màu chữ xám sáng, dễ đọc hơn */
    }
    h1, h2, h3, p, span, label, div {
        color: #cccccc !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### 📋 Nhập thông tin khách hàng")
with st.form("credit_form"):
    with st.expander("🧑 Thông tin cá nhân"):
        Name = st.text_input("Ho ten khach hang")
        gender = st.selectbox("Giới tính", options=["Chọn", "F", "M"], index=0)
        education = st.selectbox("Trình độ học vấn", options=["Chọn", "Lower secondary", "Secondary / secondary special", "Academic degree", "Incomplete higher", "Higher education"])
        birth_date = st.date_input("Ngày sinh", value=date(1990, 1, 1))
        days_employed = st.number_input("Số ngày làm việc", min_value=-1000000, max_value=365243, step=1)
        income_type = st.selectbox("Loại thu nhập", ['Working', 'Unemployed', 'Student', 'State servant', 'Pensioner', 'Maternity leave', 'Commercial associate', 'Businessman'])

    with st.expander("💳 Thông tin tín dụng"):
        external_1 = st.number_input("Điểm XHTD tổ chức A", min_value=0.0, max_value=1.0, format="%.5f")
        external_2 = st.number_input("Điểm XHTD tổ chức B", min_value=0.0, max_value=1.0, format="%.5f")
        external_3 = st.number_input("Điểm XHTD tổ chức C", min_value=0.0, max_value=1.0, format="%.5f")
        amt_credit = st.number_input("Hạn mức vay hiện tại", min_value=0.0, step=1e5)
        amt_annuity = st.number_input("Khoản trả góp hàng tháng", min_value=0.0, step=1e5)
        amt_debt = st.number_input("Tổng dư nợ hiện tại", min_value=0.0, step=1e6)
        amt_credit_sum = st.number_input("Tổng hạn mức tín dụng được cấp", min_value=0.0, step=1e6)
        MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M = st.number_input("Tỷ lệ sử dụng hạn mức 3 tháng gần nhất", min_value=0.0, max_value=1.0)

    with st.expander("📅 Lịch sử tín dụng"):
        change_date = st.date_input("Ngày thay đổi CMND")
        application_date = st.date_input("Ngày nộp đơn vay")
        credit_durations_input = st.text_area("Thời gian còn lại các khoản vay (ngày, cách nhau bằng dấu phẩy)", "30, 60, 90")
        credit_application_date = st.date_input("Ngày đăng ký tín dụng")

    with st.expander("🧾 Giấy tờ & phương tiện"):
        flag_document_3 = st.radio("Có tài liệu số 3?", ("Có", "Không"))
        flag_own_car = st.radio("Có sở hữu xe hơi?", ("Có", "Không"))

    submit = st.form_submit_button("Dự đoán điểm tín dụng")

    if submit:
        credit_durations = [int(x.strip()) for x in credit_durations_input.split(',') if x.strip().isdigit()]

        features = {
            'AVG_EXT_SOURCE': np.mean([external_1, external_2, external_3]),
            'EXT_SOURCE_1': external_1,
            'EXT_SOURCE_2': external_2,
            'EXT_SOURCE_3': external_3,
            'TERM': round(amt_credit / amt_annuity) if amt_annuity else np.nan,
            'DAYS_BIRTH': compute_days_birth(birth_date),
            'DEBT_CREDIT_RATIO': compute_debt_credit_ratio(amt_debt, amt_credit_sum),
            'CODE_GENDER': encode_gender(gender) if gender != "Chọn" else np.nan,
            'NAME_EDUCATION_TYPE': encode_education(education if education != "Chọn" else None),
            'DAYS_EMPLOYED': encode_days_employed(days_employed),
            'MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M': MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M,
            'AMT_ANNUITY': amt_annuity,
            'DAYS_ID_PUBLISH': calculate_days_id_publish(change_date.strftime('%Y-%m-%d'), application_date.strftime('%Y-%m-%d')),
            'AMT_EARLY_SUM_SUM_ALL': 0,
            'DAYS_CREDIT_ENDDATE_max': calculate_days_credit_enddate_max(credit_durations),
            'NAME_INCOME_TYPE': map_income_type(income_type),
            'FLAG_DOCUMENT_3': convert_document_3(flag_document_3),
            'FLAG_OWN_CAR': convert_car_ownership(flag_own_car),
            'DAYS_CREDIT_max': calculate_days_credit(credit_application_date),
            'NAME_PRODUCT_TYPE_street_sum': 0
        }

        X_input = pd.DataFrame([features])
        X_input = X_input.astype({"TERM": float, "AMT_ANNUITY": float})
        score = model.predict(X_input)[0]
        scaled_score = round(score * 100, 2)
        credit_limit = suggest_credit_limit(scaled_score)

        st.markdown(f"""
        <div>
            <h2>🎯 Điểm tín dụng: <span>{scaled_score}/100</span></h2>
            <h3 >💰 Hạn mức vay gợi ý: <span>{credit_limit:,.0f} VNĐ</span></h3>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🧠 Giải thích mô hình")
        st.write("Biểu đồ dưới đây cho thấy các đặc trưng ảnh hưởng nhất đến điểm tín dụng cua khach hang:")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap.Explanation(values=shap_values,
                                        base_values=explainer.expected_value,
                                        data=X_input,
                                        feature_names=X_input.columns),
                       show=False)
        st.pyplot(fig)
