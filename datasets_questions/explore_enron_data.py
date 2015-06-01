#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint
import math

pp = pprint.PrettyPrinter()


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)

print enron_data["MARTIN AMANDA K"]
for k in enron_data["MARTIN AMANDA K"]:
    print k + ": " + str(enron_data["MARTIN AMANDA K"][k])
print len(enron_data["MARTIN AMANDA K"])

pois = [person for (person,feature) in enron_data.items() if feature["poi"] == True]
print len(pois)

print enron_data["PRENTICE JAMES"]["total_stock_value"]

print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

for k in enron_data.keys():
    if "LAY" in k:
        print k + " got " + str(enron_data[k]["total_payments"])
for k in enron_data.keys():
    if "SKILLING" in k:
        print k + " got " + str(enron_data[k]["total_payments"])
for k in enron_data.keys():
    if "FASTOW" in k:
        print k + " got " + str(enron_data[k]["total_payments"])


print len([person for (person,feature) in enron_data.items() if feature["salary"] != "NaN" ])
print len([person for (person,feature) in enron_data.items() if feature["email_address"] != "NaN" ])

ll = len(enron_data)
notp = len([person for (person,feature) in enron_data.items() if feature["total_payments"] == "NaN" ])
print float(float(notp)/float(ll) * 100)

llpoi = len([person for (person,feature) in enron_data.items() if feature["poi"] == True ])
notppoi = len([person for (person,feature) in enron_data.items() if (feature["poi"] == True) & (feature["total_payments"] == "NaN") ])
print llpoi
print notppoi
print float(float(notppoi)/float(llpoi) * 100)
