#!/usr/bin/python

"""Skeleton code for k-means clustering mini-project."""


import pickle
# import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
from operator import itemgetter
from sklearn import preprocessing


def draw(pred, features, poi, mark_poi=False, name="image.png",
         f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters."""
    # ## plot each cluster with a different color--add more colors for
    # ## drawing more than five clusters
    colors = ["b", "r", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # ##place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r",
                            marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# ## load in the dict of dicts containing all data on each person in dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl",
                             "r"))

# ## there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

# ## the input features we want to use
# ## can be any key in the person-level dict (salary, director_fees, etc.)
feature_1 = "from_messages"  # "salary"
feature_2 = 'salary'  # "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

# Make up the first point to answer the quiz question
# finance_features[0] = [200000., 1000000.]

# perform feature scaling on data
min_max_scaler = preprocessing.MinMaxScaler()
finance_features = min_max_scaler.fit_transform(finance_features)

# return scaled first item for the quiz question
# print('First item in finance_features: '.format(finance_features[0]))

# ## in the "clustering with 3 features" part of the mini-project,
# ## you'll want to change this line to
# ## for f1, f2, _ in finance_features:
# ## (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

# ## cluster here; create predictions of the cluster labels
# ## for the data and store them to a list called pred

# init, fit, and predict pred
pred = KMeans(n_clusters=2).fit_predict(data)

# Find the max and min exercised_stock_options values
people_list = list(data_dict.keys())
info = []
for i in people_list:
    if data_dict[i]["salary"] != "NaN":  # ignore "NaN" values
        info.append([i, data_dict[i]["salary"]])  # build
    # sort on exercised_stock_options (and put the largest values first)
    info = sorted(info, key=itemgetter(1), reverse=True)

# ## rename the "name" parameter when you change the number of features
# ## so that the figure gets saved to a different file
try:
    draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf",
         f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")
