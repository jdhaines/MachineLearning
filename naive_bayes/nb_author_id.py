#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ## features_train and features_test are the features for the training
# ## and testing datasets, respectively
# ## labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
clf = GaussianNB()  # init GaussianNB class on variable clf

t0 = time()  # start time to train
clf.fit(features_train, labels_train)  # fit the classifier
print("training time: ", round(time()-t0, 3), "s")  # output time to fit

t1 = time()  # start time to predict
pred = clf.predict(features_test)  # run the prediction
print("prediction time: ", round(time()-t1, 3), "s")  # output time to predict

accuracy = accuracy_score(labels_test, pred)  # calculate the accuracy

# print("Pred is ", pred, "Accuracy is ", accuracy)  # test print line
# ~ 1.2 seconds to train, ~ .3 seconds to predict, ~ 97.4% accuracy

"""
Notes from Josh:

First we have to make a new instance of the GaussianNB class.  Then we have to
fit the data set into the classifier.  The data we want are about 90% of your
total data set.  We use the features and labels of most of the data set to fit
our classifier.  This is essentially a stored mapping of how the data looks.

Then we use this to make predictions.  We set this up with the predict method
and only the 10% remaining data set.  We input the features from this remaining
data and try to predict the labels.  "pred" is essentially an array of our
predicted labels using our fitted classifier.

If we did our job correctly, "pred" should match "labels_test".  In other words
the predicted labels on our 10% holdout data should match the labels from that
data.  Checking our work is the reason for holding back some of our data.

The accuracy check is simply going value by value through our arrays and
comparing our predicted labels with the actual labels.  This accuracy score
will show us how close our model fits our data.

"""
#########################################################
