import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from time import sleep

import pandas as pd
import numpy as np
import os



st.set_page_config(
    page_icon="๐ฅ",
    page_title="medical cost prediction",
    initial_sidebar_state="expanded",
    layout="wide",
)
    
X_train = pd.read_csv('X_train.csv').iloc[: , 1:]
X_test = pd.read_csv('X_test.csv').iloc[: , 1:]

y_train = pd.read_csv('y_train.csv').iloc[: , 1:]
y_test = pd.read_csv('y_test.csv').iloc[: , 1:]
# x1; ๊ธฐ์กด, x2; ๋ฒ์ฃผํbmi, x3; ๋ฒ์ฃผํbmi, smoker interaction์ถ๊ฐ
# x1 vs x2; bmi, ์ซ์ํ vs ๋ฒ์ฃผํ์ ํ
# x2 vs x3; bmi, smoker interaction ์ถ๊ฐํจ๊ณผ

# y1; ๊ธฐ์กด, y2; ์๋ฃ๋น์์10%
# y1 vs y2; ์๋ฃ๋น์์10% ๋์ ์์ธกํ  ๋ ์ด๋ค๊ฒ ์ฑ๋ฅ์ด ๋ ์ข์์ง

X_train1 = X_train.drop(
    ['bmi_over30','bmi_smoker','bmi_smoker2'],
    axis=1
    )
X_train2 = X_train.drop(['bmi','bmi_smoker','bmi_smoker2'],axis=1)
X_train3 = X_train.drop(['bmi','bmi_smoker2'],axis=1)

y_train1 = y_train['charges']
y_train2 = y_train['charges2']


X_test1 = X_test.drop(
    ['bmi_over30','bmi_smoker','bmi_smoker2'],
    axis=1
    )
X_test2 = X_test.drop(['bmi','bmi_smoker','bmi_smoker2'],axis=1)
X_test3 = X_test.drop(['bmi','bmi_smoker2'],axis=1)

y_test1 = y_test['charges']
y_test2 = y_test['charges2']

from sklearn.linear_model import LinearRegression

X_train_now = X_train1
y_train_now = y_train1

linear_regressor = LinearRegression()
linear_regressor.fit(X_train_now,y_train_now)



st.header("Medical Cost predictor")
st.subheader("๋์ ์ ๋ณด๋ฅผ ์๋ ฅํ๊ณ , ํฅํ ํ์ํ ์๋ฃ๋น๋ฅผ ์์ธกํด๋ณด์ธ์!!๐")

col1, col2 = st.columns([1, 2])

with col1:
    sex = st.radio(
        "์ฑ๋ณ์ ์ ํํด์ฃผ์ธ์",
        ('๋จ์','์ฌ์'))

    text_input_2 = st.text_input(
    "๋์ด๋ฅผ ์๋ ฅํด์ฃผ์ธ์(๋จ์:์ธ) ๐","40"
    )

    text_input_3 = st.text_input(
    "ํค๋ฅผ ์๋ ฅํด์ฃผ์ธ์(๋จ์:cm) ๐","164"
    )
    text_input_4 = st.text_input(
        "๋ชธ๋ฌด๊ฒ๋ฅผ ์๋ ฅํด์ฃผ์ธ์(๋จ์:kg) ๐","55"
    )
    if sex=='๋จ์': 
        gender=1
    else:
        gender=0
    age = int(text_input_2)
    heights=int(text_input_3)/100
    weights=int(text_input_4)
    bmi=round(weights/(heights**2),2)
    st.write("๋น์ ์ BMI์ง์:",bmi)
    smoking = st.radio(
        "๐ฌํก์ฐ ์ฌ๋ถ๋ฅผ ์ ํํ์ธ์",
        ('ํก์ฐ','๋นํก์ฐ'))
    if smoking =='ํก์ฐ':
        smoker=1
    else:
        smoker=0
    if bmi>=30:
        bmi_over30 =1
    else: bmi_over30=0

    bmi_smoker = bmi + smoker

    children = st.selectbox(
        '์๋์ ์๋ ๋ช๋ช์๋๊น?',
        ('0','1','2','3','4','5'))
    child=int(children)



with col2:

    st.subheader("๋น์ ์ ๋ฏธ๋ ์์ ์๋ฃ๋น๋???")
    input_data = (age,gender,bmi,child,smoker,1,0,0,0)
    input_data_np=np.array(input_data)
    input_data_reshape = input_data_np.reshape(1,-1)

    y_pred = linear_regressor.predict(input_data_reshape)
    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred[0],0),"USD ์๋๋ค.")

    st.subheader("1kg ๊ฐ๋์ ์๋ฃ๋น ์ ๊ฐ์ก")
    weights2=int(text_input_4)-1
    bmi2=round(weights2/(heights**2),2)
    if bmi2>=30:
        bmi_over30_2 =1
    else: bmi_over30_2=0
    bmi_smoker2 = bmi2 + smoker

    input_data2 = (age,gender,bmi2,child,smoker,1,0,0,0)
    input_data_np2=np.array(input_data2)
    input_data_reshape2 = input_data_np2.reshape(1,-1)

    y_pred2 = linear_regressor.predict(input_data_reshape2)

    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred[0]-y_pred2[0],0),"USD ์๋๋ค.")

    st.subheader("๊ธ์ฐ์ ์๋ฃ๋น ์ ๊ฐ์ก")

    input_data3 = (age,gender,bmi,child,0,1,0,0,0)
    input_data_np3=np.array(input_data3)
    input_data_reshape3 = input_data_np3.reshape(1,-1)

    y_pred3 = linear_regressor.predict(input_data_reshape3)

    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred[0]-y_pred3[0],0),"USD ์๋๋ค.")

    st.subheader("์๋ 1๋ช ๋ ๋ณ์์ ์ถ๊ฐ๋ก ์ค๋นํ  ์๋ฃ๋น")
    child2 = child + 1
    input_data4 = (age,gender,bmi,child2,smoker,1,0,0,0)
    input_data_np4=np.array(input_data4)
    input_data_reshape4 = input_data_np4.reshape(1,-1)

    y_pred4 = linear_regressor.predict(input_data_reshape4)

    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred4[0]-y_pred[0],0),"USD ์๋๋ค.")

