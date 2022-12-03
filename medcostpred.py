import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from time import sleep

import pandas as pd
import numpy as np
import os



st.set_page_config(
    page_icon="🏥",
    page_title="medical cost prediction",
    initial_sidebar_state="expanded",
    layout="wide",
)
    
X_train = pd.read_csv('X_train.csv').iloc[: , 1:]
X_test = pd.read_csv('X_test.csv').iloc[: , 1:]

y_train = pd.read_csv('y_train.csv').iloc[: , 1:]
y_test = pd.read_csv('y_test.csv').iloc[: , 1:]
# x1; 기존, x2; 범주형bmi, x3; 범주형bmi, smoker interaction추가
# x1 vs x2; bmi, 숫자형 vs 범주형전환
# x2 vs x3; bmi, smoker interaction 추가효과

# y1; 기존, y2; 의료비상위10%
# y1 vs y2; 의료비상위10% 대상 예측할 때 어떤게 성능이 더 좋을지

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



st.header("나의 정보를 입력하시고, 향후 의료비를 확인하세요!")

col1, col2 = st.columns([1, 2])

with col1:
    sex = st.radio(
        "성별을 선택해주세요",
        ('남자 ','여자'))

    text_input_2 = st.text_input(
    "나이를 입력해주세요(단위:세) 👇","40"
    )

    text_input_3 = st.text_input(
    "키를 입력해주세요(단위:cm) 👇","164"
    )
    text_input_4 = st.text_input(
        "몸무게를 입력해주세요(단위:kg) 👇","55"
    )
    if sex=="남자": 
        gender=1
    else:
        gender=0
    age = int(text_input_2)
    heights=int(text_input_3)/100
    weights=int(text_input_4)
    bmi=round(weights/(heights**2),2)
    st.write("당신의 BMI지수:",bmi)
    smoking = st.radio(
        "🚬흡연 여부를 선택하세요",
        ('흡연','비흡연'))
    if smoking =='흡연':
        smoker=1
    else:
        smoker=0
    if bmi>=30:
        bmi_over30 =1
    else: bmi_over30=0

    bmi_smoker = bmi + smoker

    st.subheader("당신의 미래 예상 의료비는???")
    input_data = (age,gender,bmi,1,smoker,1,0,0,0)
    input_data_np=np.array(input_data)
    input_data_reshape = input_data_np.reshape(1,-1)

    y_pred = linear_regressor.predict(input_data_reshape)
    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred[0],0),"USD 입니다.")

with col2:
    st.subheader("1kg 감량시 의료비 절감액")
    weights2=int(text_input_4)-1
    bmi2=round(weights2/(heights**2),2)
    if bmi2>=30:
        bmi_over30_2 =1
    else: bmi_over30_2=0
    bmi_smoker2 = bmi2 + smoker

    input_data2 = (age,gender,bmi2,1,smoker,1,0,0,0)
    input_data_np2=np.array(input_data2)
    input_data_reshape2 = input_data_np2.reshape(1,-1)

    y_pred2 = linear_regressor.predict(input_data_reshape2)

    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred[0]-y_pred2[0],0),"USD 입니다.")

    st.subheader("금연시 의료비 절감액")

    input_data3 = (age,gender,bmi2,1,0,1,0,0,0)
    input_data_np3=np.array(input_data3)
    input_data_reshape3 = input_data_np3.reshape(1,-1)

    y_pred3 = linear_regressor.predict(input_data_reshape3)

    #print(np.sqrt(mean_squared_error(y_test1, y_pred)), r2_score(y_test1, y_pred))
    st.write(round(y_pred[0]-y_pred3[0],0),"USD 입니다.")

