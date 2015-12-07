#!/usr/bin/python
"""Docstring here."""


def outlier_cleaner(predictions, ages, net_worths):
    """
    Function to clean up the outliers.

    Clean away the 10% of points that have the largest
    residual errors (difference between the prediction
    and the actual net worth).

    Return a list of tuples named cleaned_data where
    each tuple is of the form (age, net_worth, error).
    """
    # import numpy as np  # not used
    from operator import itemgetter

    cleaned_data = []
    # ## your code goes here

    # convert the parameters from nd.numpy arrays to lists
    predictions.tolist()
    ages.tolist()
    net_worths.tolist()

    # build the cleaned_data list (still has all points)
    cleaned_data = zip(ages, net_worths, abs(predictions - net_worths))

    # sort the list on the error term (largest first)
    cleaned_data = sorted(cleaned_data, key=itemgetter(2), reverse=False)

    # slice off the first 10 items
    # if you wanted to remove more or less outliers, you'd have to change the
    # "9" in the line below... probably not the best way to do this.
    # print map(itemgetter(-1), cleaned_data)  # print error value (debugging)
    for i in range(9):
        cleaned_data.pop()

    # print map(itemgetter(-1), cleaned_data)  # print error value (debugging)

    # check your work (should be 81)
    print('Length of "cleaned_data" is: {}.'.format(len(cleaned_data)))
    return cleaned_data
