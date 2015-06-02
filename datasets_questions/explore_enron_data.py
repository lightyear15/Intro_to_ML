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

print "length of the dataset: " + str(len(enron_data))
print "\n\n"

print "item list for MARTIN AMANDA K" 
print enron_data["MARTIN AMANDA K"]
for k in enron_data["MARTIN AMANDA K"]:
    print k + ": " + str(enron_data["MARTIN AMANDA K"][k])
print len(enron_data["MARTIN AMANDA K"])
print "\n\n"

pois = [person for (person,feature) in enron_data.items() if feature["poi"] == True]
print "number of pois: " + str(len(pois))
print "\n\n"

print "prentice tot stock: " + str(enron_data["PRENTICE JAMES"]["total_stock_value"])
print "\n\n"

print "colwell mails to poi count: " + str(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print "\n\n"

print "skilling ex stock options: " + str(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])
print "\n\n"

for k in enron_data.keys():
    if "LAY" in k:
        print k + " got " + str(enron_data[k]["total_payments"])
for k in enron_data.keys():
    if "SKILLING" in k:
        print k + " got " + str(enron_data[k]["total_payments"])
for k in enron_data.keys():
    if "FASTOW" in k:
        print k + " got " + str(enron_data[k]["total_payments"])
print "\n\n"

print "salary entries missing: " + str(len([person for (person,feature) in enron_data.items() if feature["salary"] != "NaN" ]))
print "mail address entries missing: " + str(len([person for (person,feature) in enron_data.items() if feature["email_address"] != "NaN"]))
print "\n\n"

ll = len(enron_data)
notp = len([person for (person,feature) in enron_data.items() if feature["total_payments"] == "NaN" ])
print "tot_payments entries missing: " + str(notp)
print "percentage tot_payments entries missing on total: " + str(float(float(notp)/float(ll) * 100))
print "\n\n"

llpoi = len([person for (person,feature) in enron_data.items() if feature["poi"] == True ])
notppoi = len([person for (person,feature) in enron_data.items() if (feature["poi"] == True) & (feature["total_payments"] == "NaN") ])
print "pois: " + str(llpoi)
print "pecentage tot_pay entries missing on poi count: " + str(float(float(notppoi)/float(llpoi) * 100))
print "\n\n"

