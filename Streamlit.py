import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, date
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load mô hình LightGBM
model = lgb.Booster(model_file='final_model.txt')

# ---------------------- Hàm xử lý ----------------------
def encode_education(level):
    mapping = {
        "Trung học cơ sở": 0,
        "Trung học phổ thông / Trung cấp": 1,
        "Đại học chưa hoàn thành": 2,
        "Đại học": 3,
        "Sau đại học": 4
    }
    return mapping.get(level, np.nan)

def encode_gender(gender):
    return 0 if gender == "Nữ" else 1 if gender == "Nam" else np.nan

def encode_days_employed(days):
    return days if days != 365243 else np.nan

def compute_debt_credit_ratio(debt, credit_sum):
    return debt / credit_sum if credit_sum != 0 else 0

def compute_days_birth(birth_date):
    return (birth_date - datetime.today().date()).days

def calculate_days_id_publish(change_date_str, application_date_str):
    change_date = datetime.strptime(change_date_str, '%Y-%m-%d')
    application_date = datetime.strptime(application_date_str, '%Y-%m-%d')
    return (application_date - change_date).days * -1

def calculate_days_credit(application_date):
    return (application_date - datetime.today().date()).days

def calculate_days_credit_enddate_max(credit_durations):
    return max(credit_durations) * -1 if credit_durations else np.nan

def map_income_type(income_type):
    mapping = {
        'Làm công ăn lương': '0', 'Thất nghiệp': '1', 'Sinh viên': '2',
        'Công chức nhà nước': '3', 'Người nghỉ hưu': '4', 'Nghỉ thai sản': '5',
        'Đối tác kinh doanh': '6', 'Chủ doanh nghiệp': '7'
    }
    return float(mapping.get(income_type, np.nan))

def convert_document_3(value):
    return 1 if value == 'Có' else 0

def convert_car_ownership(value):
    return 1 if value == 'Có' else 0


def score_scaling(p):
    pdo = 50
    base_score = 500

    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(50))
    score = offset + factor * np.log((1 - p) / p)
    return score

def suggest_credit_limit(score_100, max_amount=500_000_000):
    if score_100 >= 80:
        return 0.8 * max_amount
    elif score_100 >= 60:
        return 0.6 * max_amount
    elif score_100 >= 40:
        return 0.4 * max_amount
    else:
        return 0.2 * max_amount

# ---------------------- Giao diện người dùng ----------------------
st.set_page_config(page_title="Credit Scoring App", layout="centered")
st.title("💳 Ứng dụng chấm điểm tín dụng khách hàng")

#page_bg_img = '''
#<style>
#.stApp {
#background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
#background-size: cover;
#background-repeat: no-repeat;
#background-attachment: fixed;
#}
#</style>
#'''

# Chèn CSS vào app
#st.markdown(page_bg_img, unsafe_allow_html=True)

# 1. Thông tin cá nhân
with st.expander("👤 Thông tin cá nhân"):
    col1, col2 = st.columns([2, 2])
    with col1:
        Name = st.text_input("Họ tên KH:")
    with col2:
        cmnd = st.text_input("Số CMND/CCCD")

    col3, col4 = st.columns([2, 2])
    with col3:
        gender = st.selectbox("Giới tính", options=["Chọn", "Nữ", "Nam"], index=0)
    with col4:
        birth_date = st.date_input("Ngày sinh",min_value=date(1960, 1, 1),max_value=date(2010, 12, 31))

    col5, col6 = st.columns([2, 2])
    with col5:
        change_date_cmnd = st.date_input("Ngày cấp CMND", value=date.today())
    with col6:
        education = st.selectbox("Trình độ học vấn", options=["Chọn", "Trung học cơ sở", "Trung học phổ thông / Trung cấp", "Đại học", "Đại học chưa hoàn thành", "Sau đại học"])

    col7, col8 = st.columns([2, 2])
    with col7:
        days_employed = st.number_input("Số ngày làm việc", min_value=-1000000, max_value=365243, step=1)
    with col8:
        income_type = st.selectbox("Loại thu nhập", ['Chọn', 'Làm công ăn lương', 'Thất nghiệp', 'Sinh viên', 'Công chức nhà nước', 'Người nghỉ hưu', 'Nghỉ thai sản', 'Đối tác kinh doanh', 'Chủ doanh nghiệp'])
# 5. Giấy tờ bổ sung
with st.expander("📎 Hồ sơ khoản vay"):
    checklist_options = [
        "Đăng ký kết hôn",
        "Sổ hộ khẩu / Giấy tạm trú",
        "Giấy xác nhận tình trạng hôn nhân",
        "Đăng ký xe ô tô / Giấy tờ xe",
        "Giấy chứng nhận quyền sử dụng đất (Sổ đỏ/sổ hồng)",
        "Hợp đồng lao động",
        "Sao kê lương 3 tháng gần nhất",
        "Giấy đăng ký kinh doanh",
        "Báo cáo thuế/Doanh thu",
        "Hợp đồng cho thuê tài sản",
        "Giấy tờ sở hữu tài sản cho thuê",
        "Hợp đồng mua bán nhà/xe",
        "Dự toán chi phí xây dựng/sửa nhà",
        "Hóa đơn học phí / hợp đồng du học",
        "Sổ tiết kiệm / tài sản đảm bảo khác",
        "Ảnh chụp tài sản đảm bảo",
        "Hợp đồng tín dụng cũ (nếu có)",
        "Giấy xác nhận thu nhập bổ sung",
        "Hợp đồng thế chấp tài sản",
        "Khác (tài liệu bổ sung)"
    ]

    # Chia thành 2 hàng
    half = len(checklist_options) // 2
    row1 = checklist_options[:half]
    row2 = checklist_options[half:]

    document_flags = {}
    uploaded_files = {}

    def render_row(docs):
        cols = st.columns(2)
        for idx, doc in enumerate(docs):
            with cols[idx % 2]:
                checked = st.checkbox(doc, key=f"check_{doc}")
                document_flags[doc] = int(checked)

                if checked:
                    uploaded_file = st.file_uploader(
                        f"Tải lên file cho: {doc}",
                        key=f"upload_{doc}",
                        type=["pdf", "jpg", "png", "jpeg", "docx"]
                    )
                    uploaded_files[doc] = uploaded_file
                else:
                    uploaded_files[doc] = None

    #st.markdown("### 📄 Hàng 1:")
    render_row(row1)

    #st.markdown("### 📄 Hàng 2:")
    render_row(row2)

    # Ví dụ flag
    flag_document_3 = document_flags.get("Sổ hộ khẩu / Giấy tạm trú", 0)
    flag_own_car = document_flags.get("Đăng ký xe ô tô / Giấy tờ xe", 0)

# 3. Thông tin tín dụng
with st.expander("💰 Thông tin tín dụng"):
    col1, col2 = st.columns([2, 2])
    with col1:
        external_1 = st.number_input("Điểm XHTD KH tại CIC", min_value=0.0, format="%.5f")
    with col2:
        external_2 = st.number_input("Điểm XHTD KH tại PCB", min_value=0.0, format="%.5f")

    col3, col4 = st.columns([2, 2])
    with col3:
        external_3 = st.number_input("Điểm XHTD KH tại Viettel", min_value=0.0, format="%.5f")
    with col4:
        amt_credit = st.number_input("Hạn mức vay hiện tại (triệu VNĐ)", min_value=0.0, step=0.1)

    col5, col6 = st.columns([2, 2])
    with col5:
        amt_annuity = st.number_input("Khoản trả góp hàng tháng (triệu VNĐ)", min_value=0.0, step=0.1)
    with col6:
        amt_debt = st.number_input("Tổng dư nợ hiện tại (triệu VNĐ)", min_value=0.0, step=0.1)

    col7, col8 = st.columns([2, 2])
    with col7:
        amt_credit_sum = st.number_input("Tổng hạn mức tín dụng được cấp (triệu VNĐ)", min_value=0.0, step=0.1)
    with col8:
        MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M = st.number_input("Tỷ lệ sử dụng hạn mức 3 tháng gần nhất", min_value=0.0, max_value=1.0)

# 4. Lịch sử tín dụng (danh sách khoản vay)
#st.markdown("### 🗓 Lịch sử tín dụng (các khoản vay còn hiệu lực)")
with st.expander("📄 Thông tin các khoản vay"):
    if "loan_entries" not in st.session_state:
        st.session_state.loan_entries = []

    col1, col2 = st.columns([2, 1])
    with col1:
        new_loan_id = st.text_input("🔹 Mã khoản vay", key="loan_id")
    with col2:
        new_days_remaining = st.number_input("🔸 Số ngày còn lại (ngày)", min_value=1, max_value=5000, step=1, key="days_remaining")

    if st.button("➕ Thêm khoản vay"):
        if new_loan_id:
            st.session_state.loan_entries.append({
                "Mã khoản vay": new_loan_id,
                "Số ngày còn lại": new_days_remaining
            })

    if st.session_state.loan_entries:
        st.dataframe(pd.DataFrame(st.session_state.loan_entries))

# 6. Submit & xử lý mô hình
submit = st.button("🚀 Chấm điểm tín dụng")
if submit:
    if income_type == 'Chọn':
        st.warning("Vui lòng chọn loại thu nhập")
    else:
        credit_durations = [entry["Số ngày còn lại"] for entry in st.session_state.loan_entries]

        features = {
            'AVG_EXT_SOURCE': np.mean([external_1, external_2, external_3]),
            'EXT_SOURCE_1': external_1,
            'EXT_SOURCE_2': external_2,
            'EXT_SOURCE_3': external_3,
            'TERM': round((amt_credit * 1e6) / (amt_annuity * 1e6)) if amt_annuity else np.nan,
            'DAYS_BIRTH': compute_days_birth(birth_date),
            'DEBT_CREDIT_RATIO': compute_debt_credit_ratio(amt_debt, amt_credit_sum),
            'CODE_GENDER': encode_gender(gender) if gender != "Chọn" else np.nan,
            'NAME_EDUCATION_TYPE': encode_education(education if education != "Chọn" else None),
            'DAYS_EMPLOYED': encode_days_employed(days_employed),
            'MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M': MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M,
            'AMT_ANNUITY': amt_annuity * 1e6,
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
        scaled_score = round(score_scaling(score), 2)
        credit_limit = suggest_credit_limit(scaled_score)

        st.markdown(f"""
        <div>
            <h2>🎯 Điểm tín dụng: <span>{scaled_score}/100</span></h2>
            <h3 >💰 Hạn mức vay gợi ý: <span>{credit_limit:,.0f} VNĐ</span></h3>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🧠 Giải thích mô hình")
        st.write("Biểu đồ dưới đây cho thấy các đặc trưng ảnh hưởng nhất đến điểm tín dụng của khách hàng:")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap.Explanation(values=shap_values,
                                        base_values=explainer.expected_value,
                                        data=X_input,
                                        feature_names=X_input.columns),
                       show=False)
        st.pyplot(fig)