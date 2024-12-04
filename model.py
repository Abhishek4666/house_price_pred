import streamlit as st
import pickle

st.set_page_config(page_title="House price prediction",
page_icon=" :money_with_wings:",layout="wide")

st.title(":heavy_dollar_sign: House price prediction :moneybag:")

model1 = pickle.load(open('gbr.pkl','rb'))
model2 = pickle.load(open('adbr.pkl','rb'))

c1,c2=st.columns(2)
n1 = c1.number_input("Income")
n2 = c2.number_input("House age")

n3 = c1.number_input("No. of rooms")
n4 = c2.number_input("No. of bedrooms")

n5 = c1.number_input("Block population")
n6 = c2.number_input("Household occupancy")

n7 = c1.number_input("Lattitude")
n8 = c2.number_input("Longitude")

new_feature = [[n1,n2,n3,n4,n5,n6,n7,n8]]

c3,c4=st.columns(2)

if c3.button("GBR Model prediction"):
	t1 = model1.predict(new_feature)
	c3.subheader(t1)

if c4.button("ADR Model prediction"):
	t2 = model2.predict(new_feature)
	c4.subheader(t2)

