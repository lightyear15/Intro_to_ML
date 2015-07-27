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
data_frame = pd.DataFrame(index=data_dict.keys(), columns=data_dict["METTS MARK"].keys())
for name in data_dict.keys():
    data_frame.loc[name] = pd.Series(data_dict[name])

# replacing NaN values    
data_frame.replace("NaN", np.nan, inplace=True)
data_frame["email_address"].replace(np.nan, "", inplace=True)
data_frame.index.name = "full_name"
data_frame = data_frame.reset_index()

# removing those entries which have very few values
thre = 0.30
person_classes = { "lessthre":0, "less2thre": 0, "less3thre": 0, "more3thre":0}
for person_index in data_frame.index:
    person = data_frame.iloc[person_index]    
    ratio = float(person.count()) / float(len(person))
    #print person["full_name"] + " ratio: " + str(ratio)
    if ratio < thre:
        person_classes["lessthre"] += 1
    elif ratio < 2*thre:
        person_classes["less2thre"] += 1
    elif ratio < 3*thre:
        person_classes["less3thre"] += 1
    else:
        person_classes["more3thre"] += 1

print "employees count: "
print person_classes

toRemove = []
for person_index in data_frame.index:    
    person = data_frame.iloc[person_index]    
    ratio = float(person.count()) / float(len(person))
    if ratio < thre:
        toRemove.append(person_index)
        print "removing " + person["full_name"]
    
data_frame.drop(toRemove, axis= 0, inplace = True)
        
feature_classes = { "lessthre":0, "less2thre": 0, "less3thre": 0, "more3thre":0}
for feature in data_frame.columns:    
    ratio = float(data_frame[feature].count()) / float(len(data_frame[feature]))    
    if ratio < thre:
        feature_classes["lessthre"] += 1
    elif ratio < 2*thre:
        feature_classes["less2thre"] += 1
    elif ratio < 3*thre:
        feature_classes["less3thre"] += 1
    else:
        feature_classes["more3thre"] += 1
print "features count"
print feature_classes

toRemove = []
for feature in data_frame.columns:    
    ratio = float(data_frame[feature].count()) / float(len(data_frame[feature]))
    if ratio < thre:
        toRemove.append(feature)
        print "removing " + feature
    
data_frame.drop(toRemove, axis= 1, inplace = True)



### Task 2: Remove outliers
# remove outliers from finance data
finance_feature = ["salary", "bonus", "total_stock_value", "expenses", 
                   "exercised_stock_options", "other", "restricted_stock"]
finance_base_feature = "total_payments"
outlier_list = {}

from sklearn import linear_model
from scipy import stats


for f in finance_feature:
    tmp_frame = data_frame.loc[~((data_frame[finance_base_feature].isnull()) | (data_frame[f].isnull()))]
    tmp_frame.reset_index(inplace=True)
    X = np.reshape( np.array(tmp_frame[finance_base_feature]), (len(tmp_frame[finance_base_feature]), 1))
    Y = np.reshape( np.array(tmp_frame[f]), (len(tmp_frame[f]), 1))
    reg = linear_model.LinearRegression()
    reg.fit (X, Y)
    pred = reg.predict(X)
    error = abs(pred - Y)
    threshold = stats.scoreatpercentile(error, 90)
    for i in range(0,len(error)):
        if error[i] > threshold:
            name = tmp_frame["full_name"].loc[i]
            if name in outlier_list.keys():
                outlier_list[name] += 1
            else:
                outlier_list[name] = 1            
for i in range(0, len(finance_feature)+1):
    cc = sum(np.array(outlier_list.values()) == i)
    print "outliers on " + str(i) + " features: " + str(cc)
print "total outliers: " + str(len(outlier_list.keys()))

for name in outlier_list.keys():
    if outlier_list[name] >= 4:
        print "removing: " + name
        data_frame = data_frame.loc[data_frame["full_name"] != name]

# removing outliers from email data
email_feature_list = (("from_messages", ["from_this_person_to_poi"]), 
                     (("to_messages"), ["from_poi_to_this_person", "shared_receipt_with_poi"]))
outlier_list = {}
for feature_set in email_feature_list:
    base_f = feature_set[0]
    for f in feature_set[1]:        
        tmp_frame = data_frame.loc[~((data_frame[base_f].isnull()) | (data_frame[f].isnull()))]
        tmp_frame.reset_index(inplace=True)
        X = np.reshape( np.array(tmp_frame[base_f]), (len(tmp_frame[base_f]), 1))
        Y = np.reshape( np.array(tmp_frame[f]), (len(tmp_frame[f]), 1))
        reg = linear_model.LinearRegression()
        reg.fit (X, Y)
        pred = reg.predict(X)
        error = abs(pred - Y)
        threshold = stats.scoreatpercentile(error, 90)
        for i in range(0,len(error)):
            if error[i] > threshold:
                name = tmp_frame["full_name"].loc[i]
                if name in outlier_list.keys():
                    outlier_list[name] += 1
                else:
                    outlier_list[name] = 1            
for i in range(0, 4):
    cc = sum(np.array(outlier_list.values()) == i)        
    print "outliers on " + str(i) + " features: " + str(cc)
print "total outliers: " + str(len(outlier_list.keys()))
print "total dataset length: " + str(len(data_frame))

for name in outlier_list.keys():
    if outlier_list[name] >= 2:
        print "removing: " + name
        data_frame = data_frame.loc[data_frame["full_name"] != name]
print "final dataset length: " + str(len(data_frame))

print "after outlier removal"
print "non-pois people count: " + str(data_frame["poi"].loc[data_frame["poi"]==False].count())
print "pois people count: " + str(data_frame["poi"].loc[data_frame["poi"]==True].count())
print "\n\n"


# feature selection by decision tree
df = data_frame.fillna(0.0)
df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
X = np.array(df.values)
Y = np.array(data_frame["poi"].values)
dt = ensemble.ExtraTreesClassifier(min_samples_split=10)
dt.fit(X, Y)
d = dict(zip(df.columns.values, dt.feature_importances_))
feature_toRemove = []
for feature in d.keys():
    print feature + " importance: " + str(d[feature])
    if d[feature] < 0.01:
        feature_toRemove.append(feature)
print "feature to remove " + str(feature_toRemove)
data_frame.drop(feature_toRemove, axis=1, inplace=True)





### Task 3: Create new feature(s)
# create new feature from finance data
FEATURE_THRESHOLD = 0.1
important_feature = []

for feature in d.keys():
    if d[feature] > FEATURE_THRESHOLD:
        important_feature.append(feature)
for i in range(0, len(important_feature)):
    ifeature = important_feature[i]
    en_feature = ifeature + str("^2")
    data_frame[en_feature] = data_frame[ifeature] **2;    
    en_feature = "log_" + ifeature
    data_frame[en_feature] = data_frame[ifeature].apply(np.log);
    for j in range(i+1, len(important_feature)):
        jfeature = important_feature[j]
        en_feature = ifeature + "*" + jfeature
        data_frame[en_feature] = data_frame[ifeature] * data_frame[jfeature];

df = data_frame.fillna(0.0)
df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
X = np.array(df.values)
Y = np.array(data_frame["poi"].values)
dt = ensemble.ExtraTreesClassifier(min_samples_split=10)
dt.fit(X, Y)
d = dict(zip(df.columns.values, dt.feature_importances_))
feature_toRemove = []
for feature in d.keys():
    print feature + " importance: " + str(d[feature])
    if d[feature] < 0.01:
        feature_toRemove.append(feature)
print "feature to remove " + str(feature_toRemove)
data_frame.drop(feature_toRemove, axis=1, inplace=True)


# analyzing mails this step may take long time

ATTEMPTS = 50
PERSON_SUBSET = 20
MAIL_PER_PERSON = 200
important_word_dict = {}
poi_ratio = float(data_frame["poi"].loc[data_frame["poi"]==True].count()) / float(data_frame["poi"].count())

for i in range(0,ATTEMPTS):
    if i % 10 == 0:
        print "attempt " + str(i)
    poi_person = data_frame.loc[data_frame["poi"]==True]    
    rnd_poi_person = poi_person.loc[np.random.choice(poi_person.index, int(round(poi_ratio*PERSON_SUBSET)))]
    nonpoi_person = data_frame.loc[data_frame["poi"]==False]    
    rnd_nonpoi_person = nonpoi_person.loc[np.random.choice(nonpoi_person.index, int(round((1-poi_ratio)*PERSON_SUBSET)))]
    sub_d_frame = pd.concat([rnd_poi_person, rnd_nonpoi_person])
    sub_d_frame.reset_index(inplace = True)
    
    email_list = []
    poi_labels = []
    for name in sub_d_frame["full_name"].values:
        person = EnronEmployee(name, sub_d_frame)
        tmp1, tmp2 = person.analyzeMails(MAIL_PER_PERSON/2, MAIL_PER_PERSON/2)
        email_list += tmp1
        poi_labels += tmp2
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.25, stop_words="english")
    features_train = vectorizer.fit_transform (email_list)
    dt = ensemble.ExtraTreesClassifier()
    dt.fit(features_train.toarray(), poi_labels)
  
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]    
    sumw = 0
    idx = 0

    while sumw < 0.9 and idx < 1000:
        word = vectorizer.get_feature_names()[indices[i]]
        score = importances[indices[i]]
        
        if word in important_word_dict:
            important_word_dict[word] += score / ATTEMPTS
        else:
            important_word_dict[word] = score / ATTEMPTS
        sumw += importances[indices[i]]
        idx += 1

print important_word_dict

# computing the tfidf factor for the top 10 important words
#warning this may take very long time
import operator
import pickle

important_word_dict = pickle.load( open("important_word_pickle.pkl", "r"))
data_frame = pickle.load( open("data_frame_1.pkl", "r") )

sorted_wlist = sorted(important_word_dict.items(), key=operator.itemgetter(1), reverse=True)
vocabulary = {}
for i in range(0,10):
    k = sorted_wlist[i]    
    vocabulary[k[0]] = i
    data_frame[k[0]] = 0.0

for entry in range(0,len(data_frame)):
    person = EnronEmployee(data_frame["full_name"].iloc[entry], data_frame)
    mails = person.vectorizeMails(vocabulary)
    vect_words = person.vectorizeMails(vocabulary)
    for word in vect_words.keys():
        data_frame[word].loc[entry] = vect_words[word]


# analyzing the added features
df = data_frame.fillna(0.0)
df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
X = np.array(df.values)
Y = np.array(data_frame["poi"].values)
dt = ensemble.ExtraTreesClassifier(min_samples_split=10)
dt.fit(X, Y)
d = dict(zip(df.columns.values, dt.feature_importances_))
feature_toRemove = []
for feature in d.keys():
    print feature + " importance: " + str(d[feature])
    if d[feature] < 0.01:
        feature_toRemove.append(feature)
print "feature to remove " + str(feature_toRemove)
data_frame.drop(feature_toRemove, axis=1, inplace=True)



### Store to my_dataset for easy export below.
my_dataset = pdFrame2Dict(data_frame)
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )

tmp = list(data_frame.columns.values)
tmp.remove("full_name")
tmp.remove("email_address")
tmp.remove("poi")
feature_list = ["poi"]
feature_list += tmp
pickle.dump(feature_list, open("my_feature_list.pkl", "w") )



### Extract features and labels from dataset for local testing
matrix  = featureFormat(my_dataset, feature_list);
labels, features = targetFeatureSplit(matrix)
features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels)

### Task 4: Try a varity of classifiers



pca_params = {"pca__n_components" : [10, 12, 15, 17]}
tree_params = {"tree__min_samples_split" : [2, 3, 5, 7, 10]}
rndft_params = {"rndft__n_estimators" : [2, 3, 5, 7, 15, 20, 50]}
svc_params = {"svc__C" : [0.5, 1, 5, 10, 100]}
adabst_params = {"adabst__n_estimators" : [5, 10, 15, 20, 50]}


tmpclf = Pipeline([ ("pca", PCA()), ("tree", tree.DecisionTreeClassifier())])
tmp = pca_params.copy()
tmp.update(tree_params)
res = GridSearchCV(tmpclf, tmp, scoring="f1")
print "decisionTree result"
res.fit(features_train, labels_train)
res.score(features_test, labels_test)
print res.best_score_
BEST_SCORE = res.best_score_
clf = Pipeline([ ("pca", PCA(res.best_params_["pca__n_components"])), 
                ("tree", tree.DecisionTreeClassifier(res.best_params_["tree__min_samples_split"]))])
#test_classifier(res, my_dataset,feature_list)

clf = Pipeline([("pca", PCA()), 
                ("rndft", ensemble.RandomForestClassifier())])
tmp = pca_params.copy()
tmp.update(rndft_params)
res = GridSearchCV(clf, tmp, scoring="f1")
print "Random forest result"
res.fit(features_train, labels_train)
res.score(features_test, labels_test)
print res.best_score_
if BEST_SCORE < res.best_score_:
    clf = Pipeline([ ("pca", PCA(res.best_params_["pca__n_components"])), 
                ("rndft", ensemble.RandomForestClassifier(res.best_params_["rndft__n_estimators"]))])
    BEST_SCORE = res.best_score_
#test_classifier(res, my_dataset,feature_list)
#test_classifier(res, my_dataset,feature_list)

clf = Pipeline([ ("scale", preprocessing.MinMaxScaler()),("pca", PCA()), ("svc", LinearSVC())])
tmp = pca_params.copy()
tmp.update(svc_params)
res = GridSearchCV(clf, tmp, scoring="f1")
print "SVC result"
res.fit(features_train, labels_train)
res.score(features_test, labels_test)
print res.best_score_
if BEST_SCORE < res.best_score_:
    clf = Pipeline([ ("pca", PCA(res.best_params_["pca__n_components"])), 
                ("svc", LinearSVC(C=res.best_params_["svc__C"]))])
    BEST_SCORE = res.best_score_


clf = Pipeline([("pca", PCA()), ("adabst", ensemble.AdaBoostClassifier())])
tmp = pca_params.copy()
tmp.update(adabst_params)
res = GridSearchCV(clf, tmp, scoring="f1")
print "adaboost result"
res.fit(features_train, labels_train)
res.score(features_test, labels_test)
print res.best_score_
if BEST_SCORE < res.best_score_:
    clf = Pipeline([ ("pca", PCA(res.best_params_["pca__n_components"])), 
                ("adabst", ensemble.AdaBoostClassifier(res.best_params_["adabst__n_estimators"]))])
    BEST_SCORE = res.best_score_

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, feature_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)