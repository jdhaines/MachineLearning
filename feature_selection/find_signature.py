#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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
importances = sorted(clf.feature_importances_, reverse=True)
important_features = importances[:4]
# important_features = [importances[x] for x in importances if importances[x] > .2]
print('Top features: {}'.format(important_features))

# Most important feature
top_feature = max(importances)
print('Top single feature is: {}'.format(top_feature))

# plot what's happening
# plt.scatter(features_train, labels_train, color='blue', label='train')
# plt.scatter(features_test, labels_test, color='red', label='test')
# plt.scatter(features_test, pred, color='green', label='prediction')
# plt.title('Feature Selection')
# plt.show()
