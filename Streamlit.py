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


def compute_age_exact(birth_date):  # change
    today = datetime.today().date()

    # Số năm đầy đủ
    years = today.year - birth_date.year

    # Nếu chưa đến sinh nhật năm nay thì trừ 1 năm
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1

    # Tính số ngày từ sinh nhật năm nay đến hôm nay
    last_birthday = birth_date.replace(year=today.year)
    if last_birthday > today:
        last_birthday = last_birthday.replace(year=today.year - 1)

    days_since_birthday = (today - last_birthday).days
    # Tính tổng ngày trong năm (năm hiện tại)
    next_birthday = last_birthday.replace(year=last_birthday.year + 1)
    days_in_year = (next_birthday - last_birthday).days

    # Tuổi chính xác có phần thập phân
    age = years + days_since_birthday / days_in_year

    return round(age, 1)
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

def calculate_max_utilization(df):
    # Kiểm tra sự tồn tại của cột "Ngày trả theo lịch"
    if "Ngày trả theo lịch" not in df.columns:
        st.warning("Không tìm thấy cột 'Ngày trả theo lịch' trong dữ liệu.")
        return 0

    # Đảm bảo các giá trị là kiểu datetime
    df["Ngày trả theo lịch"] = pd.to_datetime(df["Ngày trả theo lịch"], errors='coerce')

    # Tạo cột 'Month'
    df['Month'] = df['Ngày trả theo lịch'].apply(lambda x: x.strftime('%Y-%m') if pd.notnull(x) else None)

    # Lọc khoản vay thẻ
    cc_df = df[df['Khoản vay thẻ?']]

    if cc_df.empty:
        return 0

    # Tính toán tỷ lệ sử dụng hạn mức
    cc_df['Utilization_Ratio'] = cc_df['Dư nợ'] / cc_df['Hạn mức được cấp']
    avg_utilization_per_month = cc_df.groupby('Month')['Utilization_Ratio'].mean()
    max_avg_utilization_3m = avg_utilization_per_month.tail(3).max()

    return max_avg_utilization_3m

def calculate_street_loan_count(df):
    return df[df['Kênh'] == 'Tại quầy'].shape[0]

def score_scaling(p):
    pdo = 50
    base_score = 500

    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(50))
    score = offset + factor * np.log((1 - p) / p)
    return score

def suggest_credit_limit(score, age, gender, education, avg_limit_3m, utilization_rate, #change
                         low, med, high, cap):
    """
    Hàm đề xuất hạn mức vay dựa trên nhiều yếu tố đầu vào.

    Tham số:
        - score: điểm tín dụng
        - age: tuổi
        - gender: 'Nam' hoặc 'Nữ'
        - education: trình độ học vấn (string)
        - avg_limit_3m: hạn mức bình quân 3 tháng gần nhất
        - utilization_rate: tỷ lệ sử dụng hạn mức hiện tại (0–1)
        - low, med, high: các ngưỡng điểm tín dụng phân loại
        - cap: giá trị trần hạn mức cho phép

    Trả về:
        - hạn mức đề xuất (làm tròn đến hàng chục ngàn)
    """

    # 1. Hệ số theo điểm tín dụng
    if score <= low:
        score_factor = 0
    elif score <= med:
        score_factor = 0.75
    elif score <= high:
        score_factor = 1.5
    else:
        score_factor = 2

    # 2. Hệ số theo tuổi
    if age < 25:
        age_factor = 0.8
    elif age <= 35:
        age_factor = 1.0
    else:
        age_factor = 1.2

    # 3. Hệ số giới tính
    if gender.lower() == 'Nữ':
        gender_factor = 1.0
    elif gender.lower() == 'Chọn':
        gender_factor = 0.5
    else:
        gender_factor = 0.95


    # 4. Hệ số học vấn
    edu_factors = {
        "Trung học cơ sở": 0.7,
        "Trung học phổ thông / Trung cấp": 0.85,
        "Đại học chưa hoàn thành": 0.95,
        "Đại học": 1.0,
        "Sau đại học": 1.2
    }
    edu_factor = edu_factors.get(education, 1.0)

    # 5. Hệ số tỷ lệ sử dụng hạn mức
    if utilization_rate > 0.9:
        utilization_factor = 0.6
    elif utilization_rate > 0.6:
        utilization_factor = 0.8
    elif utilization_rate > 0.3:
        utilization_factor = 1.0
    else:
        utilization_factor = 1.2

    # Tổng hệ số nhân
    total_factor = (score_factor *
                    age_factor *
                    gender_factor *
                    edu_factor *
                    utilization_factor)

    # Hạn mức đề xuất, giới hạn bởi cap
    suggested_limit = min(avg_limit_3m * total_factor, cap)

    return round(suggested_limit, -4)  # Làm tròn theo hàng chục ngàn

# ---------------------- Giao diện người dùng ----------------------
st.set_page_config(page_title="Credit Scoring App", layout="wide")
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
    col1, col2 = st.columns([1, 1])
    with col1:
        Name = st.text_input("Họ tên KH:")
        gender = st.selectbox("Giới tính", options=["Chọn", "Nữ", "Nam"], index=0)
        change_date = st.date_input("Ngày cấp CMND", min_value=date(1980, 1, 1),max_value=date(2025, 12, 31),value=date.today())
        employment_status = st.selectbox("Tình trạng việc làm", ["Đang làm việc", "Thất nghiệp", "Nghỉ hưu", "Khác"],
                                         key="employment_status")
        if employment_status == "Đang làm việc":
            years_worked = st.number_input("Số năm làm việc", min_value=0, max_value=100, step=1, key="years_worked")
            days_employed = years_worked * 365
        else:
            days_employed = 1

    with col2:
        cmnd = st.text_input("Số CMND/CCCD")
        birth_date = st.date_input("Ngày sinh", min_value=date(1960, 1, 1), max_value=date(2010, 12, 31))
        education = st.selectbox("Trình độ học vấn",
                                 options=["Chọn", "Trung học cơ sở", "Trung học phổ thông / Trung cấp", "Đại học",
                                          "Đại học chưa hoàn thành", "Sau đại học"])
        income_type = st.selectbox("Loại thu nhập",
                                   ['Chọn', 'Làm công ăn lương', 'Thất nghiệp', 'Sinh viên', 'Công chức nhà nước',
                                    'Người nghỉ hưu', 'Nghỉ thai sản', 'Đối tác kinh doanh', 'Chủ doanh nghiệp'])

    st.markdown("---")

    col3, col4 = st.columns([1, 1])
    with col3:
        application_date = st.date_input("Ngày nộp đơn vay",min_value=date(2025, 1, 1),max_value=date(2025, 12, 31),value=date.today())
    with col4:
        credit_application_date = st.date_input("Ngày mở khoản vay đầu tiên (ở tất cả TCTD)",min_value=date(1980, 1, 1),max_value=date(2025, 12, 31),value=date.today())

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
        DPD = st.selectbox("Khách hàng có đang quá hạn không?", ["Có", "Không"], key="DPD")


# 4. Lịch sử tín dụng (danh sách khoản vay)
#st.markdown("### 🗓 Lịch sử tín dụng (các khoản vay còn hiệu lực)")
# 4. Lịch sử tín dụng (danh sách khoản vay)
with st.expander("📄 Thông tin các khoản vay"):
    if "loan_entries" not in st.session_state:
        st.session_state.loan_entries = []

    # Tạo các cột cho tất cả các input trong một hàng
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    with col1:
        new_loan_id = st.text_input("🔹 Mã khoản vay", key="loan_id")
    with col2:
        new_days_remaining = st.number_input("🔸 Số ngày còn lại", min_value=1, max_value=5000, step=1, key="days_remaining")
    with col3:
        new_limit = st.number_input("Hạn mức được cấp", min_value=0, step=1000000, format="%d", key="limit")
    with col4:
        new_outstanding = st.number_input("Dư nợ", min_value=0, step=1000000, format="%d", key="outstanding")
    with col5:
        new_last_payment_date = st.date_input("Ngày trả gần nhất", key="last_payment_date", min_value=date(1980, 1, 1), max_value=date(2025, 12, 31), value=date.today())
    with col6:
        new_scheduled_payment_date = st.date_input("Ngày trả theo lịch", key="scheduled_payment_date", min_value=date(1980, 1, 1), max_value=date(2025, 12, 31), value=date.today())
    with col7:
        new_channel = st.selectbox("Kênh", ["Chọn", "Online", "Tại quầy"], key="channel")
    with col8:
        new_is_installment = st.checkbox("Trả góp", key="is_installment")
    with col9:
        new_is_cc = st.checkbox("Khoản vay thẻ?", key="is_cc")
    with col10:
        new_is_closed = st.checkbox("Đã đóng khoản vay chưa?", key="is_closed")

    # Kiểm tra và khởi tạo loan_entries nếu chưa có
    if "loan_entries" not in st.session_state:
        st.session_state.loan_entries = []

    # Định nghĩa DataFrame mặc định
    default_data = {
        "Mã khoản vay": ["N/A"],
        "Số ngày còn lại": [0],
        "Trả góp": [False],
        "Hạn mức được cấp": [0],
        "Dư nợ": [0],
        "Ngày trả gần nhất": [pd.NaT],
        "Ngày trả theo lịch": [pd.NaT],
        "Kênh": ["Không có"],
        "Khoản vay thẻ?": [False],
        "Đã đóng khoản vay chưa?": [False]
    }
    default_df = pd.DataFrame(default_data)

    # Nút thêm khoản vay
    if st.button("➕ Thêm khoản vay"):
        if new_loan_id:
            new_entry = {
                "Mã khoản vay": new_loan_id,
                "Số ngày còn lại": new_days_remaining,
                "Trả góp": new_is_installment,
                "Hạn mức được cấp": new_limit,
                "Dư nợ": new_outstanding,
                "Ngày trả gần nhất": new_last_payment_date,
                "Ngày trả theo lịch": new_scheduled_payment_date,
                "Kênh": new_channel,
                "Khoản vay thẻ?": new_is_cc,
                "Đã đóng khoản vay chưa?": new_is_closed
            }

            # Thêm dữ liệu vào session_state
            st.session_state.loan_entries.append(new_entry)
            #st.success("Đã thêm khoản vay thành công!")
        else:
            st.warning("Vui lòng điền đủ thông tin bắt buộc (Mã khoản vay và Kênh).")

        # Hiển thị DataFrame đã chuẩn hóa
        if st.session_state.loan_entries:
            st.dataframe(pd.DataFrame(st.session_state.loan_entries))


# 6. Submit & xử lý mô hình
submit = st.button("🚀 Chấm điểm tín dụng")
if submit:
    errors = []

    # Kiểm tra các selectbox bắt buộc không được để "Chọn"
    if gender == "Chọn":
        errors.append("Vui lòng chọn giới tính.")
    if education == "Chọn":
        errors.append("Vui lòng chọn trình độ học vấn.")
    if income_type == "Chọn":
        errors.append("Vui lòng chọn loại thu nhập.")
    else:
        if st.session_state.loan_entries:
            df = pd.DataFrame(st.session_state.loan_entries)
            #st.write("DataFrame hiện tại:", df)
        else:
            st.warning("Chưa có dữ liệu khoản vay.")
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
            'MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M': calculate_max_utilization(df),
            'AMT_ANNUITY': amt_annuity * 1e6,
            'DAYS_ID_PUBLISH': calculate_days_id_publish(change_date.strftime('%Y-%m-%d'), application_date.strftime('%Y-%m-%d')),
            'AMT_EARLY_SUM_SUM_ALL': sum(entry['Dư nợ'] for entry in st.session_state.loan_entries if entry['Ngày trả gần nhất'] <= entry['Ngày trả theo lịch'] and entry['Ngày trả gần nhất'] <= entry['Ngày trả theo lịch']),
            'DAYS_CREDIT_ENDDATE_max': calculate_days_credit_enddate_max(credit_durations),
            'NAME_INCOME_TYPE': map_income_type(income_type),
            'FLAG_DOCUMENT_3': convert_document_3(flag_document_3),
            'FLAG_OWN_CAR': convert_car_ownership(flag_own_car),
            'DAYS_CREDIT_max': calculate_days_credit(credit_application_date),
            'NAME_PRODUCT_TYPE_street_sum': calculate_street_loan_count(df)
        }

        X_input = pd.DataFrame([features])  # change
        X_input = X_input.astype({"TERM": float, "AMT_ANNUITY": float})
        y_pred = model.predict(X_input)[0]
        scaled_score = score_scaling(y_pred)
        age = compute_age_exact(birth_date)
        MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M = calculate_max_utilization(df)
        credit_limit = suggest_credit_limit(
            scaled_score,
            age,
            gender,
            education,
            50e+6,
            MAX_AMT_BALANCE_AMT_CREDIT_LIMIT_ACTUAL_meanonid_L3M,
            low=430.59,
            med=481.02,
            high=519.62,
            cap=500 * 1e6
        )

        st.markdown(f"""
                <div>
                    <h2>🎯 Điểm tín dụng: <span>{round(scaled_score, 2)}</span></h2>
                    <h3 >💰 Hạn mức vay gợi ý: <span>{credit_limit:,.0f} VNĐ</span></h3>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("🧠 Giải thích mô hình")
        st.write("Biểu đồ dưới đây cho thấy các đặc trưng ảnh hưởng nhất đến điểm tín dụng của khách hàng:")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(4, 2))
        shap.plots.bar(shap.Explanation(values=shap_values,
                                        base_values=explainer.expected_value,
                                        data=X_input,
                                        feature_names=X_input.columns),
                       show=False)

        st.pyplot(fig)