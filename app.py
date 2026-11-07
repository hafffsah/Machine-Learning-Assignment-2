import streamlit as st
import pandas as pd

st.title("Machine Learning Assignment 2")

dataset = st.file_uploader("STEP 1 - Upload your CSV Dataset")
if dataset is not None:
  data = pd.read_csv(dataset)
  st.session_state['data'] = data   # storing in session state to access later

st.write("STEP 2 - Model Selection")
option = st.selectbox(
    "Choose a model:",
    ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine"]
)
