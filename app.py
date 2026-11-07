import pandas as pd 
import numpy as np
import streamlit as st

from sklearn import model_selection
from sklearn import preprocessing

#   ----------------------------------------------------------------------------------------
st.title("Machine Learning Assignment 2")

st.write("STEP 1 - Upload your CSV Dataset")
dataset = st.file_uploader("Upload dataset")
if dataset is not None:
  data = pd.read_csv(dataset)
  st.session_state['data'] = data   # storing in session state to access later


if 'data' in st.session_state:
    data = st.session_state['data']
else:
    data = pd.read_csv("Churn_Modelling.csv")  # cleaned data
#   ----------------------------------------------------------------------------------------
y = data.iloc[:, -1]
x = data.iloc[:, 3:-1]
#   ----------------------------------------------------------------------------------------
encoder = preprocessing.OneHotEncoder(sparse_output= np.False_)

string_type_data = x.select_dtypes(include= ['str']).columns
encoded_data = encoder.fit_transform(x[string_type_data])

encoded_df = pd.DataFrame(
    encoded_data,
    columns= encoder.get_feature_names_out(string_type_data)
)

final_features = pd.concat(
    [x.drop(columns= string_type_data), encoded_df],
    axis=1
)
#   ----------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = model_selection.train_test_split(final_features, y, test_size= 0.2)   

scaler = preprocessing.StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#   ----------------------------------------------------------------------------------------

# ----- Model Selection and Trainings -----
st.write("")
st.write("STEP 2 - Model Selection")
model = st.selectbox(
    "Choose a model:",
    ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine"]
)
st.session_state['model'] = model
#   ----------------------------------------------------------------------------------------
from sklearn import linear_model
log_regression = linear_model.LogisticRegression()
log_regression.fit(x_train_scaled, y_train)
LR_prediction = log_regression.predict(x_test_scaled)
#   ----------------------------------------------------------------------------------------
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train_scaled, y_train)
DT_prediction = decision_tree.predict(x_test_scaled)
#   ----------------------------------------------------------------------------------------
from sklearn import neighbors
k_nn = neighbors.KNeighborsClassifier()
k_nn.fit(x_train_scaled, y_train)
KNN_prediction = k_nn.predict(x_test_scaled)
#   ----------------------------------------------------------------------------------------
from sklearn import naive_bayes
NB = naive_bayes.GaussianNB()   # not multinomial (for frequency)
NB.fit(x_train_scaled, y_train)
NB_prediction = NB.predict(x_test_scaled)
#   ----------------------------------------------------------------------------------------
from sklearn import svm
SVM = svm.SVC()
SVM.fit(x_train_scaled, y_train)
SVM_prediction = SVM.predict(x_test_scaled)
#   ----------------------------------------------------------------------------------------

# ----- Evaluation Metrics -----
from sklearn import metrics

predictions = {
    "Logistic Regression": LR_prediction, 
    "Decision Tree": DT_prediction, 
    "K-Nearest Neighbors": KNN_prediction, 
    "Naive Bayes": NB_prediction, 
    "Support Vector Machine": SVM_prediction
}

if 'model' in st.session_state:
    selected_model_pred = predictions[st.session_state['model']]
    st.session_state['accuracy'] = metrics.accuracy_score(y_test , selected_model_pred)
    st.session_state['auc'] = metrics.roc_auc_score(y_test , selected_model_pred)
    st.session_state['recall'] = metrics.recall_score(y_test , selected_model_pred)
    st.session_state['precision'] = metrics.precision_score(y_test , selected_model_pred)
    st.session_state['f1'] = metrics.f1_score(y_test , selected_model_pred)
    st.session_state['mcc'] = metrics.matthews_corrcoef(y_test , selected_model_pred)

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
    index= [st.session_state[model]],)

st.table(evaluation)
#   ----------------------------------------------------------------------------------------
