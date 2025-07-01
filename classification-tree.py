import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def train_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5
    )
    clf_gini.fit(X_train, y_train)
    return clf_gini


def train_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5
    )

    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(x_test, clf_object):
    y_pred = clf_object.predict(x_test)
    print("Predicted Values:")
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):
    print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100}%")
    print(f"Report: {classification_report(y_test, y_pred)}")


balance_data = pd.read_csv("data/balance-scale.txt")

print(f"Dataset Length: {len(balance_data)}")
print(f"Dataset Shape: {balance_data.shape}")
print(f"Dataset: {balance_data.head()}")

x = balance_data.values[:, 1:5]
y = balance_data.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=100
)

clf_gini = train_gini(X_train, y_train)
clf_entropy = train_entropy(X_train, y_train)

y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)

y_pred_entrop = prediction(X_test, clf_entropy)
cal_accuracy(y_test, y_pred_entrop)
