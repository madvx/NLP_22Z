from germansentiment import SentimentModel
import pandas as pd
import numpy as np


model = SentimentModel()

train = pd.read_csv("csv_data/train.csv")
test = pd.read_csv("csv_data/test.csv")

x = list(train["text"])
x_test = list(test["text"])
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()


pred_train = model.predict_sentiment(x)
pred_test = model.predict_sentiment(x_test)