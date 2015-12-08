#!/usr/bin/python

"""Docstring here."""

import pickle
import numpy
numpy.random.seed(42)
from sklearn import tree
from sklearn.metrics import accuracy_score

# ## The words (features) and authors (labels), already largely processed.
# ## These files should have been created from the previous (Lesson 10)
# ## mini-project.
words_file = "../text_learning/your_word_data.pkl"
# words_file = "word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
# authors_file = "email_authors.pkl"
word_data = pickle.load(open(words_file, "r"))
authors = pickle.load(open(authors_file, "r"))


# ## test_size is the percentage of events assigned to the test set (the
# ## remainder go into training)
# ## feature matrices changed to dense representations for compatibility with
# ## classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = \
    cross_validation.train_test_split(word_data, authors, test_size=0.1,
                                      random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()


# ## a classic way to overfit is to use a small number
# ## of data points and a large number of features;
# ## train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]


# ## your code goes here
# init the classifier
clf = tree.DecisionTreeClassifier()

# fit the classifier
clf.fit(features_train, labels_train)

# predict based on the features test data
pred = clf.predict(features_test)

# Get the accuracy
accuracy = accuracy_score(labels_test, pred)
print('The accuracy of the decision tree is: {}'.format(accuracy))
print('Features Train: {} and Labels Train: {}'.format(len(features_train),
                                                       len(labels_train)))

# Find the feature importances
importances = clf.feature_importances_
# important_features = sorted(importances, reverse=True)
# important_features_sliced = important_features[:4]
# print('Top features: {}'.format(important_features_sliced))

# Most important feature
top_feature = max(importances)
print('Top single feature is: {}'.format(top_feature))

# Location
location = numpy.nonzero(importances == top_feature)
print('Location of top single feature is: \
       {}'.format(importances.tolist().index(top_feature)))
print('Location = : {}'.format(location))
# giving the same values, correct answer is 33614.
# Not sure why mine is busted, I'm getting 33702.

# Get the words
# Using my number (33702) gives the correct name at least.
# I can't figure out the syntax of the get_feature_names call.
word = vectorizer.get_feature_names()[33702]
print('First word is: {}'.format(word))
