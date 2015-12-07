#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


#  ## features_train and features_test are the features for the training
#  ## and testing datasets, respectively
#  ## labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
#  features_train = features_train[:len(features_train)/100]  # 1% of data set
#  labels_train = labels_train[:len(labels_train)/100]

clf = tree.DecisionTreeClassifier(min_samples_split=40)  # call tree classifier
# print("length of features_train is ", len(features_train[0]))
t0 = time()  # start time to train
clf.fit(features_train, labels_train)  # fit the classifier
print("training time: ", round(time()-t0, 3), "s")  # output time to fit

t1 = time()  # start time to predict
pred = clf.predict(features_test)  # run the prediction
print("prediction time: ", round(time()-t1, 3), "s")  # output time to predict

accuracy = accuracy_score(labels_test, pred)  # calculate the accuracy

print("Pred is ", pred, "Accuracy is ", accuracy)  # test print line
# print("10, 25, 50", pred[10], pred[26], pred[50])  # get some specific points
# chris = 0
# for i in pred:  # Get Chri's total
#     if i == 1:
#         chris = chris + 1
# print("Chris's Total = ", chris)
# ~ 230 seconds to train, ~ 24 seconds to predict, ~ 98.4% accuracy
# C(10) = 61.6, C(100) = 61.6, C(1000) = 82.1, C(10000) = 89.2
"""
Notes from Josh:

    Nothing really to talk about here.

"""
#########################################################
