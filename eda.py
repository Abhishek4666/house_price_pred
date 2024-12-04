import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
from sklearn import metrics
import pickle

st.set_page_config(page_title="Data Analysis",page_icon=":chart_with_upwards_trend:",layout="wide")

st.title(":house: Housing data analysis :money_with_wings:")

# Loading dataset
house = datasets.fetch_california_housing(as_frame=True).frame

st.header('California Housing Price Dataset')
st.table(house.head())

st.header('California Housing Price Dataset Summary')
st.table(house.describe())

st.header('Data Visualization')
fig1 = px.imshow(house.corr(),text_auto=True,aspect='auto')
st.plotly_chart(fig1,use_container_width=True)

#Separate out feature set and target set
X = house.drop(columns='MedHouseVal')
Y = house[['MedHouseVal']]

c1,c2 = st.columns(2)
c1.subheader("Feture set")
c1.table(X.head())

c2.subheader("Target Price")
c2.table(Y.head())

#Dividing data into training and test dataset
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.25,shuffle=True,random_state=10)

c3,c4 = st.columns(2)
c3.subheader("Training Feature Set")
c3.table(xtrain.head())

c4.subheader("Training Target Set")
c4.table(ytrain.head())

c5,c6 = st.columns(2)
c5.subheader("Test Feature Set")
c5.table(xtest.head())

c6.subheader("Test Target Set")
c6.table(ytest.head())

#Regression models

gbr = GradientBoostingRegressor(max_depth=2,n_estimators=5,learning_rate=1.0)
gbr.fit(xtrain,ytrain)
ypred1 = gbr.predict(xtest)
r2_score1 = round(metrics.r2_score(ytest,ypred1),2)
model1 = pickle.dump(gbr,open('gbr.pkl','wb'))


adbr = AdaBoostRegressor()
adbr.fit(xtrain,ytrain)
ypred2 = adbr.predict(xtest)
r2_score2 = round(metrics.r2_score(ytest,ypred2),2)
model2 = pickle.dump(adbr,open('adbr.pkl','wb'))

#Comparison of models
c7,c8 = st.columns(2)
c7.subheader('Gradient Boosting Regressor model')
c7.subheader(r2_score1)

c8.subheader('Ada Boost Regressor model')
c8.subheader(r2_score2)



