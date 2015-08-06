#!/usr/bin/python


import pickle
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

from personEnron import EnronEmployee
from Giulio_aux import *

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments', 'bonus', 'expenses', 'total_stock_value', "salary", "exercised_stock_options", "restricted_stock", "shared_receipt_with_poi",
                 "other", "from_this_person_to_poi", "deferred_income", "long_term_incentive", "from_poi_to_this_person"]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
del data_dict["THE TRAVEL AGENCY IN THE PARK"]
del data_dict["TOTAL"]

# moving to pandas dataframe
data_frame = dict2PDFrame(data_dict, data_dict["METTS MARK"].keys())
pickle.dump( data_frame, open("data_frame1.pkl", "w") )


# removing those entries which have very few values
data_frame = pickle.load( open("data_frame1.pkl", "r"))
classifyEntries(data_frame, .25)
data_frame = removeClassEntries(data_frame, .25)
classifyFeatures(data_frame, .25)
data_frame = removeClassFeatures(data_frame, .25)
pickle.dump(data_frame, open("data_frame2.pkl", "w"))


### Task 2: outliers analysis
data_frame = pickle.load( open("data_frame2.pkl", "r"))
finance_feature = ["salary", "bonus", "total_stock_value", "expenses", 
                   "exercised_stock_options", "other", "restricted_stock"]
finance_base_feature = "total_payments"
findOutliers(data_frame, finance_base_feature, finance_feature)
changeOutliersValues(data_frame, finance_base_feature, finance_feature)
email_feature_list = (("from_messages", ["from_this_person_to_poi"]), 
                     (("to_messages"), ["from_poi_to_this_person", "shared_receipt_with_poi"]))
for feature_set in email_feature_list:
    findOutliers(data_frame, feature_set[0], feature_set[1])
    
for feature_set in email_feature_list:
    changeOutliersValues(data_frame, finance_base_feature, finance_feature)

pickle.dump(data_frame, open("data_frame3.pkl", "w"))


# feature selection by decision tree
FEATURE_SELECTION_PROCESS = False
data_frame = pickle.load( open("data_frame3.pkl", "r"))
if FEATURE_SELECTION_PROCESS:
    features_dict = removingFeaturePerformances(data_frame)
    print features_dict
    pickle.dump(features_dict, open("feature_selection.pkl", "w"))
features_dict = pickle.load( open("feature_selection.pkl", "r"))
blw, abv = PercentileBestFeature(features_dict, 90) 

REMOVE_FEATURES = abv    
data_frame = data_frame.drop(REMOVE_FEATURES, axis=1)
    
pickle.dump(data_frame, open("data_frame4.pkl", "w"))


### Task 3: Create new feature(s)
FEATURE_CREATION_PROCESS = False
data_frame = pickle.load( open("data_frame4.pkl", "r"))
if FEATURE_CREATION_PROCESS:
    features_dict = AddArtificialFeaturePerformance(data_frame)
    print features_dict
    pickle.dump(features_dict, open("feature_addition.pkl", "w"))

features_dict = pickle.load( open("feature_addition.pkl", "r"))
blw, abv = PercentileBestFeature(features_dict, 90) 
print blw
print abv

IMPORTANT_FEATURE = abv
data_frame = addArtificialToBestFeatures(data_frame, IMPORTANT_FEATURE)
pickle.dump(data_frame, open("data_frame5.pkl", "w"))


# analysis of emails
data_frame = pickle.load( open("data_frame5.pkl", "r"))

MAKE_MAIL_ANALYSIS = False
MAKE_EMPLOYEE_MAIL_ANALYSIS = False
if MAKE_MAIL_ANALYSIS:
    tmpWords = findBestDictionary(data_frame)
    importantWords = findBestTenWords(tmpWords)
    pickle.dump(importantWords, open("importantWords.pkl", "w"))
importantWords = pickle.load( open("importantWords.pkl", "r"))

if MAKE_EMPLOYEE_MAIL_ANALYSIS:
    dict_mail_data = retrieveEmployeeMailDict(data_frame, importantWords)
    pickle.dump(dict_mail_data, open("dict_mail_data.pkl", "w"))
dict_mail_data = pickle.load( open("dict_mail_data.pkl", "r"))
mailDFrame = mailDict2Frame(dict_mail_data)

mail_DFrame = removeWords(mailDFrame)
data_frame = data_frame.reset_index()
data_frame = data_frame.merge(mail_DFrame, how='outer', on="full_name")

data_frame.drop("index", axis=1, inplace=True)
pickle.dump(data_frame, open("data_frame6.pkl", "w"))
    

# analyze new artificial features
ARTIICIAL_FEATURE_SELECTION_PROCESS = False
data_frame = pickle.load( open("data_frame6.pkl", "r"))
if ARTIICIAL_FEATURE_SELECTION_PROCESS:
    features_dict = removingFeaturePerformances(data_frame)
    print features_dict
    pickle.dump(features_dict, open("artificial_feature_selection.pkl", "w"))
features_dict = pickle.load( open("artificial_feature_selection.pkl", "r"))
blw, abv = PercentileBestFeature(features_dict, 90) 

REMOVE_FEATURES = abv
print REMOVE_FEATURES

data_frame = data_frame.drop(REMOVE_FEATURES, axis=1)    
pickle.dump(data_frame, open("data_frame7.pkl", "w"))



### Store to my_dataset for easy export below.
data_frame = pickle.load( open("data_frame7.pkl", "r"))
data_frame = removeNans(data_frame)
my_dataset = pdFrame2Dict(data_frame)
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )

tmp = list(data_frame.columns.values)
tmp.remove("full_name")
tmp.remove("email_address")
tmp.remove("poi")
feature_list = ["poi"]
feature_list += tmp
print feature_list
pickle.dump(feature_list, open("my_feature_list.pkl", "w") )


### Extract features and labels from dataset for local testing
matrix  = featureFormat(my_dataset, feature_list);
labels, features = targetFeatureSplit(matrix)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels)

### Task 4: Try a varity of classifiers

GRIDSEARCH_ENABLE = False

if GRIDSEARCH_ENABLE:
#     pca_params = {"pca__n_components" : [10, 12, 15, 17]}
    pca_params = {"pca__n_components" : [2, 5, 10, 15, 20]}
    tree_params = {"tree__min_samples_split" : [2, 3, 5, 7, 10]}
    rndft_params = {"rndft__n_estimators" : [2, 3, 5, 7, 15, 20]}
    svc_params = {"svc__C" : [0.5, 1, 5, 10, 100]}
    adabst_params = {"adabst__n_estimators" : [5, 10, 15, 20]}
    
    
    tmpclf = Pipeline([ ("pca", PCA()), ("tree", tree.DecisionTreeClassifier())])
    tmp = pca_params.copy()
    tmp.update(tree_params)
    res = GridSearchCV(tmpclf, tmp, scoring="f1", cv=100, iid= False)
    print "decisionTree result"
    res.fit(features_train, labels_train)
    res.score(features_test, labels_test)
    print res.best_score_
    print res.best_params_
    BEST_SCORE = res.best_score_
    clf = Pipeline([ ("pca", PCA(n_components=res.best_params_["pca__n_components"])), \
                    ("tree", tree.DecisionTreeClassifier(min_samples_split=res.best_params_["tree__min_samples_split"]))])
    test_classifier(clf, my_dataset,feature_list)
    
    
    cclf = Pipeline([("pca", PCA()), 
                    ("rndft", ensemble.RandomForestClassifier())])
    tmp = pca_params.copy()
    tmp.update(rndft_params)
    res = GridSearchCV(cclf, tmp, scoring="f1")
    print "Random forest result"
    res.fit(features_train, labels_train)
    res.score(features_test, labels_test)
    print res.best_score_
    cclf = Pipeline([ ("pca", PCA(res.best_params_["pca__n_components"])), 
                    ("rndft", ensemble.RandomForestClassifier(res.best_params_["rndft__n_estimators"]))])
    test_classifier(cclf, my_dataset, feature_list)
    if BEST_SCORE < res.best_score_:
        clf = cclf
        BEST_SCORE = res.best_score_
    #test_classifier(res, my_dataset,feature_list)
    #test_classifier(res, my_dataset,feature_list)
    
    cclf = Pipeline([ ("scale", preprocessing.MinMaxScaler()),("pca", PCA()), ("svc", LinearSVC())])
    tmp = pca_params.copy()
    tmp.update(svc_params)
    res = GridSearchCV(cclf, tmp, scoring="f1")
    print "SVC result"
    res.fit(features_train, labels_train)
    res.score(features_test, labels_test)
    print res.best_score_
    cclf = Pipeline([ ("pca", PCA(res.best_params_["pca__n_components"])), 
                    ("svc", LinearSVC(C=res.best_params_["svc__C"]))])
    test_classifier(cclf, my_dataset, feature_list)
    if BEST_SCORE < res.best_score_:
        clf = cclf
        BEST_SCORE = res.best_score_
    
    
    cclf = Pipeline([("pca", PCA()), ("adabst", ensemble.AdaBoostClassifier())])
    tmp = pca_params.copy()
    tmp.update(adabst_params)
    res = GridSearchCV(cclf, tmp, scoring="f1")
    print "adaboost result"
    res.fit(features_train, labels_train)
    res.score(features_test, labels_test)
    print res.best_score_
    cclf = Pipeline([ ("pca", PCA(n_components=res.best_params_["pca__n_components"])), 
                    ("adabst", ensemble.AdaBoostClassifier(n_estimators=res.best_params_["adabst__n_estimators"]))])
    test_classifier(cclf, my_dataset, feature_list)
    if BEST_SCORE < res.best_score_:
        clf = cclf
        BEST_SCORE = res.best_score_
else:
    clf = Pipeline([ ("pca", PCA(n_components = 10)), 
                    ("tree", tree.DecisionTreeClassifier(min_samples_split= 7))])
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, feature_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
