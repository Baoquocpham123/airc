import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
file_path = r"C:\Users\Help Desk\Downloads\Monthly_data_cmo.csv"
data = pd.read_csv(file_path)


# Hiển thị dữ liệu ban đầu
print("Dữ liệu ban đầu:")
print(data.head())

# Chọn các cột số
numeric_data = data.select_dtypes(include=[np.number])

# Tính toán các số liệu thống kê chi tiết
stats = numeric_data.describe().transpose()
print("\nSố liệu thống kê chi tiết:")
print(stats)

# Vẽ biểu đồ Box Plot cho các cột số
plt.figure(figsize=(15, 10))
boxplot = numeric_data.boxplot(return_type='dict')
plt.xticks(rotation=90)  # Xoay nhãn trên trục x để dễ đọc hơn nếu cần
plt.title("Box Plot của các cột số")
plt.xlabel("Các cột")
plt.ylabel("Giá trị")

# Hiển thị các giá trị trung bình trên biểu đồ Box Plot
for column in numeric_data.columns:
    column_mean = numeric_data[column].mean()
    plt.text(numeric_data.columns.get_loc(column) + 1, column_mean, f'{column_mean:.2f}',
             horizontalalignment='center', color='red', weight='semibold')

plt.show()





# Chọn các cột số
numeric_data = data.select_dtypes(include=[np.number])

# Vẽ biểu đồ histogram cho các cột số
numeric_data.hist(bins=15, figsize=(15, 10))
plt.tight_layout()
plt.show()
