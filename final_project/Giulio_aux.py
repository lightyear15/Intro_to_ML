import numpy
import scipy
from sklearn import linear_model
import numpy as np
from numpy import dtype
import pandas as pd

def dict2PDFrame (dictio, featureList):
    dframe = pd.DataFrame(index=dictio.keys(), columns=featureList)
    for k in dictio.keys():
        for f in featureList:
            if dictio[k][f] == "NaN":
                dframe.loc[k,f] = np.nan
            else:
                dframe.loc[k,f] = dictio[k][f]
    dframe.index.name = "full_name"
    dframe = dframe.reset_index()
    return dframe


def pdFrame2Dict (data_frame):
    dictio = {}
    for idx, row in data_frame.iterrows():     
        rrow = row.drop(["full_name"])
        to_dic=rrow.to_dict()
        dictio[row["full_name"]] = to_dic
    return dictio


def classifyEntries(data_frame, thre):
    person_classes = { "lessthre":0, "less2thre": 0, "less3thre": 0, "more3thre":0}
    for person_index in data_frame.index:
        person = data_frame.iloc[person_index]    
        ratio = float(person.count()) / float(len(person))
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
    

def removeClassEntries(data_frame, thre):
    toRemove = []
    for person_index in data_frame.index:    
        person = data_frame.iloc[person_index]    
        ratio = float(person.count()) / float(len(person))
        if ratio < thre:
            toRemove.append(person_index)
            print "removing " + person["full_name"] + " who is poi: " + str (person["poi"])
    return data_frame.drop(toRemove, axis= 0)


def classifyFeatures(data_frame ,thre):
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


def removeClassFeatures(data_frame, thre):
    toRemove = []
    for feature in data_frame.columns:    
        ratio = float(data_frame[feature].count()) / float(len(data_frame[feature]))
        if ratio < thre:
            toRemove.append(feature)
            print "removing " + feature
    return data_frame.drop(toRemove, axis= 1)


from scipy import stats
def findOutliers(data_frame, base, feature_list):
    outlier_list = {}
    for f in feature_list:
        tmp_frame = data_frame.loc[~((data_frame[base].isnull()) | (data_frame[f].isnull()))]
        tmp_frame.reset_index(inplace=True)
        X = np.reshape( np.array(tmp_frame[base]), (len(tmp_frame[base]), 1))
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
    for i in range(0, len(feature_list)+1):
        cc = sum(np.array(outlier_list.values()) == i)
        print "outliers on " + str(i) + " features: " + str(cc)
    print "total outliers: " + str(len(outlier_list.keys()))



def changeOutliersValues(data_frame, base, feature_list):
    for f in feature_list:
        tmp_frame = data_frame.loc[~((data_frame[base].isnull()) | (data_frame[f].isnull()))]
        tmp_frame.reset_index(inplace=True)
        X = np.reshape( np.array(tmp_frame[base]), (len(tmp_frame[base]), 1))
        Y = np.reshape( np.array(tmp_frame[f]), (len(tmp_frame[f]), 1))
        reg = linear_model.LinearRegression()
        reg.fit (X, Y)
        pred = reg.predict(X)
        error = abs(pred - Y)
        threshold = stats.scoreatpercentile(error, 90)
        for i in range(0,len(error)):
            if error[i] > threshold:
                name = tmp_frame["full_name"].loc[i]
                data_frame[f].loc[data_frame["full_name"] == name] = pred[i]
    

from sklearn import ensemble
def featureImportance(data_frame):
    df = data_frame.fillna(0.0)
    df.replace(np.inf, 0, inplace=True)
    df.replace(-np.inf, 0, inplace=True)
    df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
    X = np.array(df.values)
    Y = np.array(data_frame["poi"].values)
    dt = ensemble.ExtraTreesClassifier(min_samples_split=2)
    dt.fit(X, Y)
    d = dict(zip(df.columns.values, dt.feature_importances_))
#     for feature in d.keys():
#         print feature + " importance: " + str(d[feature])
    return d


TEST_CYCLE = 50
from sklearn import cross_validation
from sklearn import metrics
def removingFeaturePerformances(data_frame):
    df = data_frame.fillna(0.0)
    df.replace(np.inf, 0, inplace=True)
    df.replace(-np.inf, 0, inplace=True)
    df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
    feature_list = df.columns.values
    rank = dict((el,0) for el in feature_list)
    for feature_toRemove in feature_list:        
        tmp_frame = df.copy()
        tmp_frame.drop(feature_toRemove, axis=1, inplace=True)
        X = np.array(tmp_frame.values)
        Y = np.array(data_frame["poi"].values)
        score = 0
        for i in range(0,TEST_CYCLE):
            skf = cross_validation.StratifiedKFold(Y, n_folds=18)
            for train_index, test_index in skf:
                dt = ensemble.ExtraTreesClassifier(min_samples_split=2)
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                dt.fit(X_train, Y_train)                
                score += metrics.f1_score(list(Y_test), list(dt.predict(X_test)))
        print "removing " + feature_toRemove + ", score: " + str(score)
        rank[feature_toRemove] = score
    return rank
    

def PercentileBestFeature(feature_dict, percentile):
    feat_score = feature_dict.values()
    thre = stats.scoreatpercentile(feat_score, percentile)
    below = []
    above = []
    for f in feature_dict.keys():
        if feature_dict[f] > thre:
            above.append(f)
        else:
            below.append(f)
    return below, above
    
    

from math import log
def AddArtificialFeaturePerformance(data_frame):    
    df = data_frame.fillna(0.0)
    df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
    feature_list = df.columns.values
    rank = dict((el,0) for el in feature_list)
    for f1 in range(0, len(feature_list)-1):
        for f2 in range(f1+1, len(feature_list)):
            important_feature = [feature_list[f1], feature_list[f2]]
        #print important_feature
            tmp_frame = df.copy()
            for i in range(0, len(important_feature)):
                ifeature = important_feature[i]
                en_feature = ifeature + str("^2")
                tmp_frame[en_feature] = tmp_frame[ifeature] **2;    
                en_feature = "log_" + ifeature
                tmp_frame[en_feature] = tmp_frame[ifeature].apply(np.log);
                for j in range(i+1, len(important_feature)):
                    jfeature = important_feature[j]
                    en_feature = ifeature + "*" + jfeature
                    tmp_frame[en_feature] = tmp_frame[ifeature] * tmp_frame[jfeature];
            tmp_frame = tmp_frame.fillna(0.0)
            tmp_frame.replace(np.inf, 0.0, inplace=True)
            tmp_frame.replace(-np.inf, 0.0, inplace=True)     
            X = np.array(tmp_frame.values)
            Y = np.array(data_frame["poi"].values)    
            score = 0
            for i in range(0,TEST_CYCLE):
                skf = cross_validation.StratifiedKFold(Y, n_folds=18)
                for train_index, test_index in skf:
                    dt = ensemble.ExtraTreesClassifier(min_samples_split=2)        
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]
                    dt.fit(X_train, Y_train)
                    score += metrics.f1_score(list(Y_test), list(dt.predict(X_test)))
            rank[important_feature[0]] += score
            rank[important_feature[1]] += score
            for i in rank.keys():
                print "feature: ", i, " score: ", rank[i] 
    return rank



def addArtificialToBestFeatures(data_frame, important_feature):
    df = data_frame.fillna(0.0)
    df.drop(["full_name", "poi", "email_address"], axis=1, inplace=True)
    created_data_frame = data_frame.copy()
    created_data_frame = created_data_frame.fillna(0.0)
    for i in range(0, len(important_feature)):
        ifeature = important_feature[i]
        en_feature = ifeature + str("^2")
        created_data_frame[en_feature] = created_data_frame[ifeature] **2;    
        en_feature = "log_" + ifeature
        created_data_frame[en_feature] = created_data_frame[ifeature].apply(numpy.log);
        for j in range(i+1, len(important_feature)):
            jfeature = important_feature[j]
            en_feature = ifeature + "*" + jfeature
            created_data_frame[en_feature] = created_data_frame[ifeature] * created_data_frame[jfeature]
    return created_data_frame
        

from personEnron import EnronEmployee
from sklearn.feature_extraction.text import TfidfVectorizer
def findBestDictionary(data_frame):
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
    ord_dict = sorted(important_word_dict.items(), key=operator.itemgetter(1), reverse=True)
    for i in ord_dict:
        print i[0] + " " + str(i[1])
    return important_word_dict
    

import operator
WORD_TO_REMOVE = ['7138536485', 'louis', 'joanni', 'sullivanhoueese', 'jeff', 
                  'thomson', '2819487273', '8002626000', 'kellyjohnsonenroncom',
                  "0949", "134", "31769", "5034643735", "62602", "charlen", "georgeann",
                  "johnfowlersuncom", "kennethlayenroncom", "philipp",
                  "taylorhouectect", "tim"]
def findBestTenWords(dictio):
    for w in WORD_TO_REMOVE:
        if w in dictio.keys():
            del dictio[w]
    sortDict = sorted(dictio.items(), key=operator.itemgetter(1), reverse=True)
    return sortDict   


def removeWords(mail_data_frame):
    words = mail_data_frame.columns.values
    for w in words:
        if w in WORD_TO_REMOVE:
            mail_data_frame = mail_data_frame.drop(w, axis=1)
    return mail_data_frame


import multiprocessing
from joblib import Parallel, delayed 
def parallelAux(data_frame, name, dictio):
    print name
    person = EnronEmployee(name, data_frame)
    mails = person.vectorizeMails(dictio)
    vect_words = person.vectorizeMails(dictio)
    vect_words["full_name"] = name
    return vect_words
    
def retrieveEmployeeMailDict(data_frame, dictio):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores) \
        (delayed(parallelAux)(data_frame, data_frame["full_name"].iloc[entry], dictio) for entry in range(0,len(data_frame)))
    return results


def mailDict2Frame(personMailList):
    words = []
    w = personMailList[0]
    for k in w.keys():
        if k != "full_name":
            words.append(k[0])
    lista = []
    for p in personMailList:
        dictio = {}
        for tt in p:
            if tt == "full_name":
                dictio[tt] = p[tt]
            else:
                dictio[tt[0]] = tt[1]
        lista.append(dictio)
    frame = pd.DataFrame(lista)        
    return frame


def removeNans(data_frame):
    for i in range(0, len(data_frame)):
        for c in data_frame.columns.values:
            if c == "email_address" or c=="full_name" or c=="poi":
                pass
            else:
                if numpy.isnan(data_frame[c].loc[i]):
                    data_frame[c].loc[i] = 0.0
                if numpy.isneginf(data_frame[c].loc[i]):
                    data_frame[c].loc[i] = -1000.0
                if numpy.isposinf(data_frame[c].loc[i]):
                    data_frame[c].loc[i] = +1000.0
    return data_frame