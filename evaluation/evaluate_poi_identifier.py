#!/usr/bin/python

"""
Starter code for the evaluation mini-project.

Start by copying your trained/tested POI identifier from
that which you built in the validation mini-project.

This is the second step toward building your POI identifier!

Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl",
                             "r"))

# ## add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# Split data into training and testing sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# check the provided starter code statements for accuracy
# print('Data Dict: {}'.format(data_dict))
# print('data: {}'.format(data))
# print('Labels: {}'.format(labels))
# print('Features: {}'.format(features))

# Fit a decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

# Check the accuracy
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print('Accuracy of the decision tree: {:.3f}'.format(accuracy))

# Find the number of POIs in the test set.
print('Number of POIs in the Test Set: {}'.format(sum(pred)))

# Number of people in the test set
print('Number of people in the test set: {}'.format(len(features_test)))

# Check for true positives
print
print('======================================================================')
print
print('Compare the actual test labels vs the predicted labels.'.format())
for i in range(len(pred)):
    print('Actual: {}  -  Predicted: {}'.format(labels_test[i], pred[i]))

# Find the precision and recall scores
from sklearn.metrics import recall_score, precision_score
print('Precision: {:.3f}  -  Recall: {:.3f}'.format(precision_score(
      labels_test, pred), recall_score(labels_test, pred)))

# Answer the quiz
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print('Fake Data, answering the quiz...'.format())
print('Precision: {:.3f}  -  Recall: {:.3f}'.format(precision_score(
      true_labels, predictions), recall_score(true_labels, predictions)))
# end #
