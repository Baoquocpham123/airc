import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
file_path = r"C:\Users\Help Desk\Downloads\Monthly_data_cmo.csv"
data = pd.read_csv(file_path)

# Hiển thị dữ liệu ban đầu
print("Dữ liệu ban đầu:")
print(data.head())

# Kiểm tra và xử lý các giá trị thiếu (nếu có)
data = data.dropna()

# Chọn các cột số cho PCA và LASSO (loại bỏ các cột không số)
numeric_data = data.select_dtypes(include=[np.number])

# Xác định biến mục tiêu (target value)
# Ví dụ: giả sử biến mục tiêu là cột đầu tiên
target_column = numeric_data.columns[1]
y = numeric_data[target_column]

# Loại bỏ cột mục tiêu khỏi dữ liệu đầu vào
X = numeric_data.drop(columns=[target_column])

# Tính các độ trễ (lags) cho dữ liệu chuỗi thời gian
lags = 3  # Giảm số lượng độ trễ xuống 3 thay vì 5
lagged_data = pd.concat([X.shift(lag).add_suffix(f'_lag{lag}') for lag in range(1, lags + 1)], axis=1)

# Kết hợp dữ liệu gốc với dữ liệu độ trễ
X = pd.concat([X, lagged_data], axis=1)

# Loại bỏ các hàng có giá trị NaN do tạo độ trễ
X = X.dropna().copy()
y = y[X.index]  # Đồng bộ chỉ số của y với X sau khi loại bỏ NaN

# Kiểm tra nếu dữ liệu sau khi loại bỏ NaN có còn hay không
if X.empty or y.empty:
    raise ValueError("Dữ liệu sau khi loại bỏ NaN là rỗng. Hãy kiểm tra lại việc tạo độ trễ và dữ liệu ban đầu.")

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Chuyển đổi kết quả PCA thành DataFrame để dễ dàng hình dung
X_pca_df = pd.DataFrame(data=X_pca, columns=[f'Principal Component {i+1}' for i in range(X_pca.shape[1])])

# Kết hợp kết quả PCA với biến mục tiêu để phân tích
final_df = pd.concat([X_pca_df, y.reset_index(drop=True)], axis=1)

print("\nData after applying PCA:")
print(final_df.head())

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Áp dụng hồi quy LASSO với Cross-Validation
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Đánh giá mô hình
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)

# Hiển thị kết quả
lasso_coefficients = lasso.coef_

print(f"Train R^2 score: {train_score}")
print(f"Test R^2 score: {test_score}")
print("LASSO coefficients:")
print(lasso_coefficients)

# Áp dụng TOPSIS
def topsis(df, weights, impacts):
    # Chuẩn hóa ma trận quyết định
    norm_matrix = df / np.sqrt((df ** 2).sum())

    # Áp dụng trọng số
    weighted_matrix = norm_matrix * weights

    # Xác định giải pháp lý tưởng và giải pháp tồi tệ nhất
    ideal_solution = np.max(weighted_matrix * impacts, axis=0)
    negative_ideal_solution = np.min(weighted_matrix * impacts, axis=0)

    # Tính khoảng cách đến giải pháp lý tưởng và giải pháp tồi tệ nhất
    distance_to_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))

    # Tính chỉ số tương đồng với giải pháp lý tưởng
    relative_closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

    # Xếp hạng các lựa chọn
    df['TOPSIS Score'] = relative_closeness
    df['Rank'] = df['TOPSIS Score'].rank(ascending=False)

    return df

# Áp dụng TOPSIS
criteria = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
weights = np.ones(criteria.shape[1]) / criteria.shape[1]  # Trọng số cho mỗi tiêu chí
impacts = np.ones(criteria.shape[1])  # Tất cả các tiêu chí đều có tác động tích cực

topsis_result = topsis(criteria, weights, impacts)

# Hiển thị kết quả TOPSIS

print("\nTOPSIS result:")
print(topsis_result)
