import os
import numpy as np
import sys
sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText 
from sklearn.feature_extraction.text import TfidfVectorizer


class EnronEmployee(object):
    def __init__(self, name, data_frame):
        self.name_ = name
        self.dframe_ = data_frame
        if self.getEmail():
            self.retrieveMailLists()
            self.countEmails()
        else:
            self.recvdMailCount_ = 0
            self.sentMailCount_ = 0
            self.totalMailCount_ = 0
        self.checkIfPoi()
    
    def getEmail(self):
        self.email_ = self.dframe_["email_address"].loc[self.dframe_["full_name"] == self.name_].iloc[0]
        return self.email_ is not ""
    
    def checkIfPoi(self):
        self.isPoi_ = self.dframe_["poi"].loc[self.dframe_["full_name"] == self.name_].iloc[0]
        
    def retrieveMailLists(self):
        tmp = "./emails_by_address/from_" + self.email_ + ".txt"
        if os.path.isfile(tmp):
            self.fromList_ = tmp
        else:
            self.fromList_ = ""            
        tmp = "./emails_by_address/to_" + self.email_ + ".txt"
        if os.path.isfile(tmp):
            self.toList_ = tmp
        else:
            self.toList_ = ""

    def countEmails(self):
        self.sentMailCount_ = 0
        self.recvMailCount_ = 0
        if self.fromList_ is not "": 
            mailList = open(self.fromList_,"r");
            self.sentMailCount_ = sum(1 for line in mailList)
        else:
            self.sentMailCount_ = 0
        if self.toList_ is not "":    
            mailList = open(self.toList_,"r");
            self.recvdMailCount_ = sum(1 for line in mailList)
        else:
            self.recvdMailCount_ = 0
        self.totalMailCount_ = self.sentMailCount_ + self.recvdMailCount_
    
    def getEmailCount(self):
        return self.totalMailCount_
        
    def analyzeMails(self, limit_from = float('Infinity'), limit_to = float('Infinity')):     
        if self.totalMailCount_ == 0:
            return [], []
        tmp = []
        if self.fromList_ is not "":
            if self.sentMailCount_ > limit_from:
                sentMailIdx = np.random.choice(self.sentMailCount_, limit_from)
            else:
                sentMailIdx = range(0, self.sentMailCount_)
            midx = 0
            mailList = open(self.fromList_,"r");
            for path in mailList:
                if midx in sentMailIdx:
                    path = os.path.join('..', path[:-1])
                    email = open(path, "r")
                    str_email = parseOutText(email)                              
                    tmp.append(str_email)
                midx += 1
            word_data = tmp
            poi_data = [int(self.isPoi_)] * len(tmp)
            
        word_data = tmp
        tmp = []      
        if self.toList_ is not "":
            if self.recvdMailCount_ > limit_to:
                recvdMailIdx = np.random.choice(self.recvdMailCount_, limit_to)
            else:
                recvdMailIdx = range(0, self.recvdMailCount_)
            midx = 0
            mailList = open(self.toList_,"r");
            for path in mailList:
                if midx in recvdMailIdx:
                    path = os.path.join('..', path[:-1])
                    email = open(path, "r")
                    str_email = parseOutText(email)                               
                    tmp.append(str_email)
                midx += 1               
            poi_data += [int(self.isPoi_)] * len(tmp)        
            word_data += tmp
            return word_data, poi_data
    
    def vectorizeMails(self, dictio):
        words, _ = self.analyzeMails()
        total_words = ""
        for w in words:
            total_words += " " + w
        vecter = TfidfVectorizer(sublinear_tf=True, vocabulary = dictio)
        scores = vecter.fit_transform([total_words])
        result = {}
        scores = scores.toarray()
        for i in range(0, len(scores[0])):
            result[vecter.get_feature_names()[i]] = scores[0,i]
        return result