from germansentiment import SentimentModel
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

model = SentimentModel()

train = pd.read_csv("csv_data/train.csv")
test = pd.read_csv("csv_data/test.csv")

x = list(train["text"])
x_test = list(test["text"])
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()

pred_train = []
pred_test = []

for i, xx in enumerate(x):
    if i % 100 == 0:
        print(f"{i}/{len(x)}")
    result = model.predict_sentiment([xx])[0]
    if result == "positive":
        pred_train.append(0)
    elif result == "negative":
        pred_train.append(2)
    elif result == "neutral":
        pred_train.append(1)

for i, xx in enumerate(x_test):
    if i % 100 == 0:
        print(f"{i}/{len(x_test)}")
    result = model.predict_sentiment([xx])[0]
    if result == "positive":
        pred_test.append(0)
    elif result == "negative":
        pred_test.append(2)
    elif result == "neutral":
        pred_test.append(1)

pred_train = np.array(pred_train)
pred_test = np.array(pred_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

names = ["positive", "neutral", "negative"]

print("REPORT TRAIN")
print(classification_report(y_train, pred_train, target_names=names, digits=4))
matrix = confusion_matrix(y_train, pred_train)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=names)
plt.figure()
disp.plot()
plt.show()

print("REPORT TEST")
print(classification_report(y_test, pred_test, target_names=names, digits=4))
matrix = confusion_matrix(y_test, pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=names)
plt.figure()
disp.plot()
plt.show()
