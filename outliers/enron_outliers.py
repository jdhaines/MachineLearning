#!/usr/bin/python
"""Docstring here."""

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat
from operator import itemgetter
# from feature_format import targetFeatureSplit


# ## read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl",
                             "r"))
# Remove the outlier.pop()
data_dict.pop("TOTAL", 0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# ## your code below

# Find the owner of the outlier
people_list = list(data_dict.keys())
info = []
for i in people_list:
    if data_dict[i]["salary"] != "NaN":  # ignore "NaN" values
        info.append([i, data_dict[i]["salary"]])  # build the list
    # sort on bonus (and put the largest values first)
    info = sorted(info, key=itemgetter(1), reverse=True)
print(info[0])  # print only the first (highest) entry
print(info[1])  # print only the second (second highest) entry
# Once we got the oulier, we put a line above to remove it.

# Make a graphic of the data
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
