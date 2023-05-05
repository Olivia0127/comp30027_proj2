# import librart
import sklearn
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder



def process_OneHotEncoder(df,column_name:str):
    encoder = OneHotEncoder(sparse=False)
    one_hot = encoder.fit_transform(df[[column_name]])
    one_hot_df = pd.DataFrame(one_hot, columns=encoder.get_feature_names_out([column_name]))
    return one_hot_df
    
def process_OneHotEncoder_pd(train_df,test_df,column_name:str):
    one_hot_encoded_train_df = pd.get_dummies(train_df, columns=[column_name])
    one_hot_encoded_test_df = pd.get_dummies(test_df, columns=[column_name])
    return one_hot_encoded_train_df,one_hot_encoded_test_df

def docclass_preprocess(train, test, threshold):
    #change some type of class into other to decrease the dimension of matrix
    data = train.value_counts()
    data_test = test.value_counts()
    unfreq_class = []
    for cla in data.index:
        if data[cla] < threshold:
            unfreq_class.append(cla)
    for cla in data_test.index:
        if cla not in data.index:
            test = test.replace(cla, 'others')
    train = train.replace(unfreq_class, 'others')
    train.fillna('others', inplace = True)
    test = test.replace(unfreq_class, 'others')
    test.fillna('others', inplace = True)
    return train, test

