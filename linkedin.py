import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st
from PIL import Image

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame(
{"sm_li": clean_sm(s["web1h"]),
 "income": np.where(s["income"] > 9, np.nan, s["income"]),
 "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
 "parent": np.where(s["par"] == 1, 1, 0),
 "married": np.where(s["marital"] == 1, 1, 0),
 "female": np.where(s["gender"] == 2, 1, 0),
 "age": np.where(s["age"] > 98, np.nan, s["age"])}
)

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    random_state = 5)

lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
confusion_matrix(y_test, y_pred)
metrics.accuracy_score(y_test, y_pred)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

print(classification_report(y_test, y_pred))

person_1 = [8, 7, 0, 1, 1, 42]

predicted_class1 = lr.predict([person_1])
probs1 = lr.predict_proba([person_1])

image = Image.open('logo.png')

cola, colb = st.columns([1,4])
with cola:
    st.image(image)
with colb:
    st.markdown("# Predicting LinkedIn Usage")

st.markdown("### This tool uses a machine learning model to predict if a person is a LinkedIn user based six predictors.")

st.markdown("#### Select values for each predictor and view the results of model below.")

# predictor variables: income, education, parent, married, female, age

col1, col2 = st.columns(2)
with col1:
    inc = st.selectbox("Current Income Level", options = [
        "Less than $10k",
        "$10k to under $20k",
        "$20k to under $30k",
        "$30k to under $40k",
        "$40k to under $50k",
        "$50k to under $75k",
        "$75k to under $100k",
        "$100k to under $150k",
        "More than $150k"])
    
    educ = st.selectbox("Highest Level of Education", options = [
        "Less than high school", 
        "High school incomplete", 
        "High school graduate",
        "Some college, no degree",
        "Two-year associate degree from college or university",
        "Four-year college or university degree",
        "Some postgraduate or professional schooling",
        "Postgradute or professional degree"])

    par = st.selectbox("Parent of Child(ren) under 18", options = [
        "Yes",
        "No"])

if inc == "Less than $10k":
    inc = 1
elif inc == "$10k to under $20k":
    inc = 2
elif inc == "$20k to under $30k":
    inc = 3
elif inc == "$30k to under $40k":
    inc = 4
elif inc == "$40k to under $50k":
    inc = 5
elif inc == "$50k to under $75k":
    inc = 6
elif inc == "$75k to under $100k":
    inc = 7
elif inc == "$100k to under $150k":
    inc = 8
else:
    inc = 9



if educ == "Less than high school":
    educ = 1
elif educ == "High school incomplete":
    educ = 2
elif educ == "High school graduate":
    educ = 3
elif educ == "Some college, no degree":
    educ = 4
elif educ == "Two-year associate degree from college or university":
    educ = 5
elif educ == "Four-year college or university degree":
    educ = 6
elif educ == "Some postgraduate or professional schooling":
    educ = 7
else:
    educ = 8

if par == "Yes":
    par = 1
else:
    par = 0

with col2:
    mar = st.selectbox("Current Marital Status", options = [
        "Married",
        "Living with a partner",
        "Divorced",
        "Separated",
        "Widowed",
        "Never been married"
    ])

    gender = st.selectbox("Gender", options = [
        "Male",
        "Female"
    ])

    age = st.number_input("Age", 1, 98)

if mar == "Married":
    mar = 1
elif mar == "Living with a partner":
    mar = 2
elif mar == "Divored":
    mar = 3
elif mar == "Separated":
    mar = 4
elif mar == "Widowed":
    mar = 5
else:
    mar = 6

if gender == "Male":
    gender = 0
else:
    gender = 1


# Displayed Results
# predictor variables: income, education, parent, married, female, age
person_display = [inc, educ, par, mar, gender, age]

predicted_class_display = lr.predict([person_display])
probs_display = lr.predict_proba([person_display])


st.markdown("#### Model Results:")

if predicted_class_display[0] == 1:
    pred_label = "a LinkedIn user"
else:
    pred_label = "not a LinkedIn user"

prob = round(probs_display[0][1],3)

st.write(f"The individual is {pred_label}, with a probability of {prob}.")