# -*- coding: utf-8 -*-
"""data_cleaning_3.ipynb (Enhanced Version)"""

import pandas as pd
import numpy as np
import re

# ==============================================================================
# 1. ĐỌC DỮ LIỆU & ĐỔI TÊN CỘT
# ==============================================================================
try:
    # Đường dẫn file gốc của bạn
    df = pd.read_csv('../data/raw/VN_housing_dataset.csv')
    print("✅ Đã đọc file thành công!")
    print(f"Kích thước ban đầu: {df.shape}")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file csv!")
    exit()

# CẬP NHẬT: Không xóa vội 'Địa chỉ', 'Dài', 'Rộng' vì chúng ta cần dùng nó
cols_to_drop = ['Unnamed: 0', 'Ngày'] # Chỉ xóa cột không dùng được
df = df.drop(columns=cols_to_drop, errors='ignore')

# Đổi tên cột chuẩn (Thêm Length, Width)
rename_map = {
    'Quận': 'District',
    'Huyện': 'Ward',
    'Loại hình nhà ở': 'House_type',
    'Giấy tờ pháp lý': 'Legal',
    'Số tầng': 'Floors',
    'Số phòng ngủ': 'Bedrooms',
    'Diện tích': 'Area',
    'Giá/m2': 'Price_per_m2',
    'Dài': 'Length',   # Giữ lại
    'Rộng': 'Width',   # Giữ lại
    'Địa chỉ': 'Address' # Giữ lại để tách tên đường
}
df = df.rename(columns=rename_map)
print("✅ Đã đổi tên cột chuẩn.")

# ==============================================================================
# 2. XỬ LÝ SỐ LIỆU (DATA CLEANING)
# ==============================================================================
def extract_number(value):
    if pd.isna(value): return np.nan
    text = str(value).lower().replace(',', '.')
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(match.group()) if match else np.nan

# Áp dụng cho các cột số (bao gồm cả Dài, Rộng)
cols_num = ['Area', 'Price_per_m2', 'Bedrooms', 'Floors', 'Length', 'Width']
for col in cols_num:
    if col in df.columns:
        df[col] = df[col].apply(extract_number)

print("✅ Đã chuyển đổi dữ liệu sang dạng số.")

# ------------------------------------------------------------------------------
# 🌟 LOGIC MỚI: CỨU DỮ LIỆU DÀI / RỘNG (SMART IMPUTATION)
# ------------------------------------------------------------------------------
print(f"NaN trước khi xử lý: Dài={df['Length'].isna().sum()}, Rộng={df['Width'].isna().sum()}")

# 1. Nếu thiếu Rộng nhưng có Diện tích & Dài -> Tính Rộng = Area / Length
mask_w = df['Width'].isna() & df['Length'].notna() & df['Area'].notna()
df.loc[mask_w, 'Width'] = df.loc[mask_w, 'Area'] / df.loc[mask_w, 'Length']

# 2. Nếu thiếu Dài nhưng có Diện tích & Rộng -> Tính Dài = Area / Width
mask_l = df['Length'].isna() & df['Width'].notna() & df['Area'].notna()
df.loc[mask_l, 'Length'] = df.loc[mask_l, 'Area'] / df.loc[mask_l, 'Width']

# 3. Còn lại thì điền bằng Median (Trung vị) của toàn tập dữ liệu
df['Length'] = df['Length'].fillna(df['Length'].median())
df['Width'] = df['Width'].fillna(df['Width'].median())

# Điền thiếu cho các cột khác
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
df['Floors'] = df['Floors'].fillna(df['Floors'].median())
df['Legal'] = df['Legal'].fillna('Dang_cap_nhat')

print(f"NaN sau khi xử lý: Dài={df['Length'].isna().sum()}, Rộng={df['Width'].isna().sum()}")

# ------------------------------------------------------------------------------
# 🌟 LOGIC MỚI: TRÍCH XUẤT TÊN ĐƯỜNG (STREET EXTRACTION)
# ------------------------------------------------------------------------------
def get_street(addr):
    if not isinstance(addr, str): return 'Other_Street'
    # Lấy phần đầu tiên trước dấu phẩy (thường là tên đường/phố)
    return addr.split(',')[0].strip()

df['Street'] = df['Address'].apply(get_street)

# Chỉ giữ lại Top 100 đường phổ biến nhất, còn lại gộp thành 'Other_Street'
# (Để tránh tạo ra hàng nghìn cột One-Hot làm nặng máy)
top_streets = df['Street'].value_counts().nlargest(100).index
df['Street'] = df['Street'].apply(lambda x: x if x in top_streets else 'Other_Street')

print(f"✅ Đã trích xuất xong Tên đường. Số lượng đường giữ lại: {len(top_streets)}")

# ==============================================================================
# 3. TẠO BIẾN MỤC TIÊU & LỌC NHIỄU (FILTERING)
# ==============================================================================
# Tính tổng giá
df['Total_Price_Billion'] = (df['Price_per_m2'] * df['Area']) / 1000

# Xử lý text Quận/Huyện/Phường
df['District'] = df['District'].str.replace('Quận', '').str.replace('Huyện', '').str.strip()
df['Ward'] = df['Ward'].str.replace('Phường', '').str.replace('Xã', '').str.strip()
# Điền khuyết thiếu cho cột Pháp lý
df['Legal'] = df['Legal'].fillna('Dang_cap_nhat')

# Lọc dữ liệu nhiễu (Logic cũ của bạn + Bổ sung IQR Filter cho giá)
df = df.dropna(subset=['District', 'Ward']) # Xóa nếu không có địa chỉ

df = df[(df['Area'] >= 10) & (df['Area'] <= 500)]
df = df[(df['Total_Price_Billion'] >= 0.5) & (df['Total_Price_Billion'] <= 100)]

# Lọc logic phòng ở
df = df[~((df['Area'] < 40) & (df['Bedrooms'] >= 8))]
df = df[~((df['Floors'] < 2) & (df['Bedrooms'] >= 5) & (df['Area'] < 100))]

# Lọc theo IQR của đơn giá (Price_per_m2) -> Giúp loại bỏ nhà giá ảo (quá rẻ/quá đắt)
Q1 = df['Price_per_m2'].quantile(0.05)
Q3 = df['Price_per_m2'].quantile(0.95)
df = df[(df['Price_per_m2'] >= Q1) & (df['Price_per_m2'] <= Q3)]

print(f"✅ Dữ liệu sạch cuối cùng: {len(df)} dòng")

# ==============================================================================
# 4. ONE-HOT ENCODING (MÃ HÓA)
# ==============================================================================
# Danh sách các cột cần mã hóa (Bao gồm cả Street mới)
categorical_cols = ['District', 'Ward', 'House_type', 'Legal', 'Street']

# Tạo One-Hot
df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Xóa các cột không cần thiết cho việc Train (Address, Price_per_m2...)
cols_garbage = ['Address', 'Price_per_m2']
df_final = df_final.drop(columns=cols_garbage, errors='ignore')

print(f"Kích thước sau khi One-Hot: {df_final.shape}")
# Kỳ vọng số cột sẽ tăng lên khoảng 350-400 cột (do thêm Street)

# ==============================================================================
# 5. LƯU FILE
# ==============================================================================
save_path = '../data/processed/clean_vn_housing.csv'
df_final.to_csv(save_path, index=False)
print(f"✅ Đã lưu file sạch tại: {save_path}")