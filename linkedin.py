import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

person_1 = [8, 7, 0, 1, 1, 42]

predicted_class1 = lr.predict([person_1])
probs1 = lr.predict_proba([person_1])