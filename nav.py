import streamlit as st

pages = st.navigation([st.Page('eda.py',title='Housing Price Dataset'),
		       st.Page('model.py',title='Price Prediction')])

pages.run()