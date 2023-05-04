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
def process_OneHotEncoder(train_df,test_df,column_name:str):
    encoder = OneHotEncoder(sparse=False)
    one_hot_train = encoder.fit_transform(train_df[[column_name]])
    one_hot_train_df = pd.DataFrame(one_hot_train, columns=encoder.get_feature_names_out([column_name]))
    one_hot_test = encoder.fit_transform(test_df[[column_name]])
    one_hot_test_df = pd.DataFrame(one_hot_test, columns=encoder.get_feature_names_out([column_name]))
    return one_hot_train_df, one_hot_test_df
def process_OneHotEncoder_pd(train_df,test_df,column_name:str):
    one_hot_encoded_train_df = pd.get_dummies(train_df, columns=[column_name])
    one_hot_encoded_test_df = pd.get_dummies(test_df, columns=[column_name])
    return one_hot_encoded_train_df,one_hot_encoded_test_df

