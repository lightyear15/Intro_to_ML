import numpy
import scipy
from sklearn import linear_model
from numpy import dtype

import pandas as pd
def dict2PDFrame (dictio, featureList):
    """ convert dictionary formatted as 
    {name1:{feature1: value1, feature2:value2, ....}, name2:{feature1: value1, feature2:value2, ....},...}
    in a pandas dataframe formatted as
            feature1    feature2    ...
    name1   value1      value2      ...
    name2   value1      value2      ...
    
    selecting features from featureList
    """
    dframe = pd.DataFrame(index=dictio.keys(), columns=featureList)
    for k in dictio.keys():
        for f in featureList:
            if dictio[k][f] == "NaN":
                dframe.loc[k,f] = 0
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
            
        


