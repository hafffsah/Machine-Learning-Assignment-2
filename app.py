import streamlit as st
import pandas as pd

st.title("Machine Learning Assignment 2")

st.write("STEP 1 - Upload your CSV Dataset")
dataset = st.file_uploader("Upload dataset")
if dataset is not None:
  data = pd.read_csv(dataset)
  st.session_state['data'] = data   # storing in session state to access later

st.write("")
st.write("STEP 2 - Model Selection")
model = st.selectbox(
    "Choose a model:",
    ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine"]
)
st.session_state['model'] = model

st.write("")
st.write("STEP 3 - Evaluation Metrics")
evaluation = pd.DataFrame(
    {
        "Accuracy": st.session_state['accuracy'],
        "AUC Score": st.session_state['auc'], 
        "Precision": st.session_state['precision'],
        "Recall": st.session_state['recall'], 
        "F1 Score": st.session_state['f1'], 
        "MCC Score": st.session_state['mcc']
    },
    index= ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],)

