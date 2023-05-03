# import librart
import sklearn
import numpy
import pandas as pd
import pickle
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# mutualInformation selection
def MI(train_df,train_features,test_features,m):

    selector = SelectKBest(mutual_info_classif, k=m)
    selector.fit_transform(train_features, train_df["rating_label"])
    feature_idx = selector.get_support(indices=True)
    selected_features = train_features[:, feature_idx]
    selected_features_test = test_features[:, feature_idx]
    return selected_features, selected_features_test

# chi Square selection
def chi_square(train_df,train_features,test_features,m):
    selector = SelectKBest(chi2, k=m)
    selector.fit(train_features, train_df["rating_label"])
    feature_idx = selector.get_support(indices=True)
    selected_features = train_features[:, feature_idx]
    selected_features_test = test_features[:, feature_idx]
    return selected_features, selected_features_test