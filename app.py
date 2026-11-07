import streamlit as st
import pandas as pd

st.title("Machine Learning Assignment 2")

dataset = st.file_uploader("Upload your CSV Dataset")
if dataset is not None:
  df = pd.read_csv(dataset)
  st.write(df)
