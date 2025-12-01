import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
# from joblib import load
# #
ab = pd.read_csv('Database_Regressions/abalone.data', header=None)
# #
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
           'Viscera weight', 'Shell weight', 'Rings']
ab.columns = columns
ab['Sex']=ab['Sex']=ab['Sex'].map({'M':0,'F':1,'I':2})
X = ab.drop('Rings', axis=1)
# #
y = ab['Rings']
# # # -------------------------------------------------------------------------------
# # # ab.select_dtypes(include=[np.number]).plot(kind='box',figsize=(10,8))
# # # plt.title('Boxplot of Abalone Database')
# # # plt.show()
# # # -----------------------------------------------------------------------------
# # # -----------------------------------------------------------------------

# # # ---------------------------------------------------------------------------------
def remove_outlier(df, column):
# # #
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
# # #
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[(df[column] >= lower_bound) & (df[column] <= upper_bound)])
# # #
# # #
# # #
R_tdf= remove_outlier(ab,'Rings')
# #
# print(R_tdf.describe().to_string())
# # # print(ab.describe().to_string())
# # # sns.histplot(data=X,x='Length')
# # # plt.show()
# #
# #
# # # print(ab.describe())
# #
# # # ----------------------------------------------------------
# # # ------------------------------------------------------------
# # # sns.histplot(data=X,x='numeric_cols')
# # # plt.show()
# #
# #
# # # print(ab.info())
# # # plt.show()
# # # plt.subplot(1,2,1)
# # # sns.histplot(data=ab,x='Length')
# # # plt.title('Before')
# # #
# # # plt.subplot(1,2,2)
# # # sns.histplot(data=R_tdf,x='Length')
# # # plt.title('After')
# # # #
# # # plt.tight_layout()
# # # plt.show()
# #
# #
# # # print(ab.columns)
# # # print(ab.shape)
# # # print(ab.info())
# # # print(ab.describe().to_string())
# # # print(ab.isna().sum())
# # # ['M' 'F' 'I']
# #
# #
# #
# # # print(ab.head().to_string())
# # # print(ab.info())
# # # sns.histplot(data=X,x='Length')
# # # plt.show()
# #
# #
# # '''
# # ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
# #        'Viscera weight', 'Shell weight', 'Rings'],
# # '''
# #
x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42, test_size=0.3)
# print(x_train.head().to_string())
# # # print(x_test.head().to_string())
# # # print(x_train.shape)
# # # print(x_test.shape)
# # # print(y_train.head().to_string())
# # # print(50*'*')
# # # print(y_test.head().to_string())
lr = LinearRegression()
lr.fit(x_train, y_train)
# #
y_prediction = lr.predict(x_test)
# # # print(y_prediction)
# # # print(len(y_prediction))
# # # print(len(y_test))
# #
# # # MAE = mean_absolute_error(y_test, y_prediction)
MSE = mean_squared_error(y_test, y_prediction)
RMSE = np.sqrt(MSE)
# # # print(MAE)
# # # print(MSE)
# print(RMSE)
# #
# # # print(ab['Rings'].mean())
# #
# test_residual = y_prediction - y_test
#
# # # print(test_residual)
# #
# # # sns.scatterplot(x=y_test, y=test_residual)
# # # plt.axhline(y=0, color='r', linestyle='--')
# # # plt.show()
# #
final_model=LinearRegression()
final_model.fit(X.values,y)
y_hat=final_model.predict(X.values)
#
# print(y_hat)
# #
# # new_data=[[0,0.25,0.39,0.43,0.56,0.2025,0.73,0.25]]
# # print(final_model.predict(new_data))
# #
# joblib.dump(final_model,'ab(Ring).joblib')
#
# #
# final_model=joblib.load('ab(Ring).joblib')
# y_pred=final_model.predict(X)
# print(y_pred,'پیش بینی')
#
# # new_data=[[0,0.25,0.39,0.43,0.56,0.2025,0.73,0.88]]
# # print(loaded_model.predict(new_data))
# # # ----------------------------------------------------------------------
import tkinter as tk
from tkinter import messagebox
# #
# # # تابع پیش‌بینی سن
def predict_age():
    try:
        sex = int(sex_var.get())  # گرفتن انتخاب جنسیت
        length = float(length_entry.get())
        diameter = float(diameter_entry.get())
        height = float(height_entry.get())
        whole_weight = float(whole_weight_entry.get())
        shucked_weight = float(shucked_weight_entry.get())
        viscera_weight = float(viscera_weight_entry.get())
        shell_weight = float(shell_weight_entry.get())

# #         # ساخت یک آرایه ورودی برای مدل
        input_data = [[sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]]
# #
# #         # پیش‌بینی سن
        prediction = final_model.predict(input_data)
        result_label.config(text=f"پیش‌بینی سن صدف: {prediction[0]:.2f} سال")
    except ValueError:
        messagebox.showerror("خطا", "لطفاً تمام فیلدها را به درستی وارد کنید!")
# #
##ساخت پنجره Tkinter
root = tk.Tk()
root.geometry('200x600')
root.title("پیش‌بینی سن صدف (Abalone)")
# #
# # # افزودن فیلدها
tk.Label(root, text="جنسیت (0: مرد، 1: زن، 2: نامعلوم):").pack(pady=5)
sex_var = tk.StringVar(value="0")
tk.OptionMenu(root, sex_var, "0", "1", "2").pack(pady=5)
# #
tk.Label(root, text="طول (Length):").pack(pady=5)
length_entry = tk.Entry(root)
length_entry.pack(pady=5)
# #
tk.Label(root, text="قطر (Diameter):").pack(pady=5)
diameter_entry = tk.Entry(root)
diameter_entry.pack(pady=5)
# #
tk.Label(root, text="ارتفاع (Height):").pack(pady=5)
height_entry = tk.Entry(root)
height_entry.pack(pady=5)
# #
tk.Label(root, text="وزن کل (Whole Weight):").pack(pady=5)
whole_weight_entry = tk.Entry(root)
whole_weight_entry.pack(pady=5)
# #
tk.Label(root, text="وزن شل (Shucked Weight):").pack(pady=5)
shucked_weight_entry = tk.Entry(root)
shucked_weight_entry.pack(pady=5)
# #
tk.Label(root, text="وزن داخلی (Viscera Weight):").pack(pady=5)
viscera_weight_entry = tk.Entry(root)
viscera_weight_entry.pack(pady=5)
# #
tk.Label(root, text="وزن صدف (Shell Weight):").pack(pady=5)
shell_weight_entry = tk.Entry(root)
shell_weight_entry.pack(pady=5)
# #
# # # دکمه پیش‌بینی
predict_button = tk.Button(root, text="پیش‌بینی سن", command=predict_age)
predict_button.pack(pady=10)
# #
# # # برچسب برای نمایش نتیجه
result_label = tk.Label(root, text="پیش‌بینی سن صدف: ")
result_label.pack(pady=20)
# #
# # # اجرای رابط کاربری
root.mainloop()
# #
#
#
#
