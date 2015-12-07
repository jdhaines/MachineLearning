#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl",
                              "r"))
# Get number of people
print
print("Number of people is: ", len(enron_data))

# Get number of features
print
print("Number of features is: ", len(enron_data["SKILLING JEFFREY K"]))

# get number of "POI"s
print
people_list = list(enron_data.keys())
poi_count = 0
poi_list = list()  # I'll use this later
for i in people_list:
    if enron_data[i]["poi"] is True:
        poi_count += 1
        poi_list.append(i)  # easy to make this now

print("Number of POI: ", poi_count)

# Get value of James Prentice's stock.
print
print("James Prentice's Stock Value is: ", enron_data["PRENTICE JAMES"]
      ["total_stock_value"])

# Emails from Wesley Colwell
print
print("Emails from Wesley Colwell: ", enron_data["COLWELL WESLEY"]
      ["from_this_person_to_poi"])

# Stock exercised by Jeffrey Skilling
print
print("Stock Exercised by jeffrey Skilling: ", enron_data["SKILLING JEFFREY K"]
      ["exercised_stock_options"])

# Most money between CEO, Chairman, CFO?
print
print("Jeffrey Skilling (CEO) had: ", enron_data["SKILLING JEFFREY K"]
      ["total_payments"])
print("Kenneth Lay (Chairman) had: ", enron_data["LAY KENNETH L"]
      ["total_payments"])
print("Andrew Fastow (CFO) had: ", enron_data["FASTOW ANDREW S"]
      ["total_payments"])

# Get all features for one person (Skilling)
# Also shows how a blank is denoted
print
info_list = list(enron_data["SKILLING JEFFREY K"].keys())
print("Features for Jeffrey Skilling are:")
for i in info_list:
    print(i, enron_data["SKILLING JEFFREY K"][i])

# Number of people in the dataset with a known salary
print
salary_count = 0
for i in people_list:
    if enron_data[i]["salary"] != "NaN":
        salary_count += 1
print(salary_count, "People with a quantified Salary")

# Number of people with a known email address
print
email_count = 0
for i in people_list:
    if enron_data[i]["email_address"] != "NaN":
        email_count += 1
print(email_count, "People with a known email address")

# Number of people with NaN for their total payments with fraction
print
payments_missing_people_count = 0
for i in people_list:
    if enron_data[i]["total_payments"] == "NaN":
        payments_missing_people_count += 1
percent = float(payments_missing_people_count) / len(people_list) * 100
print(payments_missing_people_count, 'People with missing "Total Payments"')

# new string format style (easier...)
print('{} out of {} or {:.2f} percent.'.format(payments_missing_people_count,
                                               len(people_list), percent))

# Number of POI with NaN for their total payments with fraction
print
payments_missing_poi_count = 0
for i in poi_list:
    if enron_data[i]["total_payments"] == "NaN":
        payments_missing_poi_count += 1
percent_poi = float(payments_missing_poi_count) / len(poi_list) * 100
print('{} POIs with missing "Total Payments"'
      .format(payments_missing_poi_count))
print('{} out of {} or {:.2f} percent.'.format(payments_missing_poi_count,
      len(poi_list), percent_poi))  # new string format style (easier...)

# end
