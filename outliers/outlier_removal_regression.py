#!/usr/bin/python

"""This is a docstring."""

# import random
import numpy
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from outlier_cleaner import outlier_cleaner


# ## load up some practice data with outliers in it
ages = pickle.load(open("practice_outliers_ages.pkl", "r"))
net_worths = pickle.load(open("practice_outliers_net_worths.pkl", "r"))


# ## ages and net_worths need to be reshaped into 2D numpy arrays
# ## second argument of reshape command is a tuple of integers:
# ## (n_rows, n_columns) by convention, n_rows is the number of data points
# ## and n_columns is the number of features
ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test \
    = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

# ## fill in a regression here!  Name the regression object reg so that
# ## the plotting code below works, and you can see what your regression looks
# ## like
reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)
print("Initial slope is: {:.3f}.".format(float(reg.coef_)))  # training slope
print("Test data prediction score is: {:.3f}".format(reg.score(ages_test,
      net_worths_test)))  # score from the regression on the test data
try:
    plt.plot(ages, reg.predict(ages), color="red")
except NameError:
    pass
plt.scatter(ages_train, net_worths_train, color="red", marker="s")

# ## identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlier_cleaner(predictions, ages_train, net_worths_train)
except NameError:
    print("your regression object doesn't exist, or isn't named reg")
    print("can't make predictions to use in identifying outliers")


# ## only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages_cleaned, net_worths_cleaned, errors = zip(*cleaned_data)
    ages_cleaned = numpy.reshape(numpy.array(ages_cleaned), (len(ages_cleaned),
                                                             1))
    net_worths_cleaned = numpy.reshape(numpy.array(net_worths_cleaned),
                                                  (len(net_worths_cleaned), 1))

    # ## refit your cleaned data!
    try:
        reg.fit(ages_cleaned, net_worths_cleaned)
        plt.plot(ages_cleaned, reg.predict(ages_cleaned),
                 color="green")
        # new slope calculation with cleaned up data
        print("The new slope (green line) = {:.3f}.".format(float(reg.coef_)))
        # new score calulation with cleaned up data
        print("Test data prediction score (after cleaning) is: {:.3f}".format(
              reg.score(ages_test, net_worths_test)))
    except NameError:
        print("you don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages_cleaned, net_worths_cleaned, color="green")
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

else:
    print("outlier_cleaner() is returning an empty list, no refitting to be " +
          "done")
