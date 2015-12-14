import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file : data_dict = pickle.load(data_file)

print('Number of People in the dataset: {}'.format(len(data_dict)))
print
print('A list of the people in the dataset.')
print('{}'.format([i for i in data_dict]))
print
print("All features for a single person, let's use the CEO: Jeff Skilling.")
print('{}'.format(data_dict["SKILLING JEFFREY K"]))

print("Let's see all the salaries and number of emails sent for everyone in the dataset.")
print
print('Name -- Salary -- Emails Sent')
print
for i in sorted(data_dict):
    print('{} -- ${} -- {}'.format(i, data_dict[i]['salary'], data_dict[i]['from_messages']))

import matplotlib
from matplotlib import pyplot as plt
# %matplotlib notebook

emails_with_pois = []
is_poi = []

# make x and y axes for a graph
for i in data_dict:
    emails_with_pois.append(data_dict[i]['from_this_person_to_poi'] + data_dict[i]['from_poi_to_this_person'])
    
    # Build the is_poi list
    if data_dict[i]['poi'] == True:
        is_poi.append(1)
    else:
        is_poi.append(0)
# clean up NaN issues
emails_with_pois = [i if i != 'NaNNaN' else 0 for i in emails_with_pois]


# print('emails_with_pois: {}'.format(emails_with_pois))
# print('is_poi: {}'.format(is_poi))
fig1 = plt.figure()
plt.scatter(is_poi, emails_with_pois)
plt.xlabel('Person is a POI')
plt.ylabel('Total Emails with POIs')
plt.title('POI Status vs Total Emails with POIs')
plt.show()

# POI compensation vs emails
# compensation
from operator import add
poi_compensation = []
non_poi_compensation = []

for i in data_dict:  # Loop all people
    salary = 0
    total_stock_value = 0
    bonus = 0
    incentive = 0
    
    if data_dict[i]['poi'] == True:  # Only act if person is a POI
        if data_dict[i]['salary'] != 'NaN':
            salary = data_dict[i]['salary']
        if data_dict[i]['total_stock_value'] != 'NaN':
            total_stock_value = data_dict[i]['total_stock_value']
        if data_dict[i]['bonus'] != 'NaN':
            bonus = data_dict[i]['bonus']
        if data_dict[i]['long_term_incentive'] != 'NaN':
            incentive = data_dict[i]['long_term_incentive']
        poi_compensation.append(salary + total_stock_value + bonus + incentive)
        non_poi_compensation.append(0)  # Add zero to non_pois
    
    else:  # Person is not a POI
        if data_dict[i]['salary'] != 'NaN':
            salary = data_dict[i]['salary']
        if data_dict[i]['total_stock_value'] != 'NaN':
            total_stock_value = data_dict[i]['total_stock_value']
        if data_dict[i]['bonus'] != 'NaN':
            bonus = data_dict[i]['bonus']
        if data_dict[i]['long_term_incentive'] != 'NaN':
            incentive = data_dict[i]['long_term_incentive']
        non_poi_compensation.append(salary + total_stock_value + bonus + incentive)
        poi_compensation.append(0)  # Add zero to pois

# Emails
emails_with_pois_poi = []
emails_with_pois_non_poi = []

for i in data_dict:  # Loop all people
    if data_dict[i]['poi'] == True:  # Only act if person is a POI
        emails_with_pois_poi.append(data_dict[i]['from_this_person_to_poi'] + data_dict[i]['from_poi_to_this_person'])
        emails_with_pois_non_poi.append(0)
    else:  # Act if person is not a poi
        emails_with_pois_non_poi.append(data_dict[i]['from_this_person_to_poi'] + data_dict[i]['from_poi_to_this_person'])
        emails_with_pois_poi.append(0)
        
# clean up NaN issues
emails_with_pois_poi = [i if i != 'NaNNaN' else 0 for i in emails_with_pois_poi]
emails_with_pois_non_poi = [i if i != 'NaNNaN' else 0 for i in emails_with_pois_non_poi]

fig2 = plt.figure()
plt.scatter(poi_compensation, emails_with_pois_poi, label="POI", color="r")
plt.scatter(non_poi_compensation, emails_with_pois_non_poi, label="Non-POI", color="b")
plt.xlabel('Total Compensation')
plt.ylabel('Total Emails with POIs')
plt.title('Total Compensation vs Total Emails with POIs')
plt.legend()
plt.show()

print('Display Non-POI Compensation from largest to smallest: {}'.format(sorted(non_poi_compensation, reverse=True)))

index = [non_poi_compensation.index(607079287)]
index = index[0]
del non_poi_compensation[index]
del emails_with_pois_non_poi[index]

fig3 = plt.figure()
plt.scatter(poi_compensation, emails_with_pois_poi, label="POI", color="r")
plt.scatter(non_poi_compensation, emails_with_pois_non_poi, label="Non-POI", color="b")
plt.xlabel('Total Compensation')
plt.ylabel('Total Emails with POIs')
plt.title('Total Compensation vs Total Emails with POIs')
plt.legend()
plt.show()

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# Get list of all features
keys = data_dict['SKILLING JEFFREY K'].keys()
keys.remove('poi')  # remove poi
keys.remove('email_address')  # remove email_address
keys.insert(0, 'poi')  # put poi back in, but in the front
# print keys
features_list = ['poi',
                 # 'salary',
                 # 'to_messages',
                 # 'deferral_payments',
                 # 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 # 'restricted_stock',
                 # 'shared_receipt_with_poi',
                 # 'restricted_stock_deferred',
                 # 'total_stock_value',
                 'expenses'
                 # 'loan_advances',
                 # 'from_messages',
                 # 'other',
                 # 'from_this_person_to_poi',
                 # 'director_fees',
                 # 'deferred_income',
                 # 'long_term_incentive']
                 # 'from_poi_to_this_person'
                 ]
# list of all features available, but with 'poi' as the first.

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# from sklearn import preprocessing
# features_train_scaled = preprocessing.scale(features_train)

from sklearn.decomposition import PCA  # get access to the module
pca_test = PCA(n_components=2)  # PCA and use the best few
pca_test.fit(features_train)
# pca_test.transform(features_train)
print(pca_test.explained_variance_ratio_)

# Combination method shown at
# http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py
# pca = PCA(n_components=2)

# from sklearn.feature_selection import SelectKBest
# selection = SelectKBest(k=2)

# #  Combine PCA and KBest feature selection together
# from sklearn.pipeline import FeatureUnion
# combined_features = FeatureUnion([('pca', pca), ('univ_select', selection)])

# # Transform our data
# new_features_train = combined_features.fit(features_train_scaled, labels_train).transform(features_train_scaled)

# # Get an SVM ready
# from sklearn.svm import SVC
# clf = SVC(kernel='linear')

# Get a decision tree ready
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=45)

# # Initiate the pipeline to run the features through the svm classifier multiple times
# from sklearn.pipeline import Pipeline
# # pipeline = Pipeline([("features", combined_features), ("svm", svm)])
# pipeline = Pipeline([("features", combined_features), ("tree", tree)])

# param_grid = dict(features__pca__n_components=[1, 2], features__univ_select__k=[1, 2])  #svm__C=[0.1, 10, 100])

# from sklearn.grid_search import GridSearchCV
# clf = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
# clf.fit(features_train_scaled, labels_train)
# print(clf.best_estimator_)

# pred = clf.predict(features_test)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
accuracy = accuracy_score(labels_test, pred)
f1 = f1_score(labels_test, pred)
recall = recall_score(labels_test, pred)
precision = precision_score(labels_test, pred)

print('Our accuracy is: {}'.format(accuracy))
print('Our precision is: {}'.format(f1))
print('Our recall is: {}'.format(recall))
print('Our f1 score is: {}'.format(precision))

print(pred, labels_test)

# fig4 = plt.figure()
# plt.scatter(features_train, labels_train, label="POI", color="r")
# plt.scatter(features_test, pred, label="Predicted", color="b")
# plt.xlabel('Total Compensation')
# plt.ylabel('Total Emails with POIs')
# plt.title('Total Compensation vs Total Emails with POIs')
# plt.legend()
# plt.show()

dump_classifier_and_data(clf, my_dataset, features_list)
