import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from FeatureSelector import FeatureSelector
from Results import Results
from kFolderTester import kFolderTester

def NanCleaner(data):
    #@param data is a dataframe
    return data.apply(NanCleanerApply, axis = 0)

def NanCleanerApply(x):
    #@param x is a column of the dataset
    maskNan = pd.isna(x)
    maskNotNan = pd.notna(x)
    notNan = x[maskNotNan]
    nan = x[maskNan]
    avg = int(np.average(notNan))
    for i in range (0, len(x)):
        if(pd.isna(x[i])):
            x[i] = avg
    return x

def NanRemover(data):
    mask = data.apply(NanRemoverApply, axis = 1)
    data["mask"] = mask
    data =  data[data["mask"] == True]
    return data.iloc[:, 0:data.columns.size - 1]

def NanRemoverApply(x):
    #@param x is a row of a dataframe
    for i in range(0, len(x)):
       if( pd.isna(x[i])):
           return False
    return True

if(__name__ == "__main__"):
    np.random.seed(154)

    # Read in data and display first 5 rows
    data = pd.read_csv('glasses.csv', sep=",")

    print('The shape of our data is:', data.shape)
    #colnames = ["RI","Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]
    #one hot encoding: transorming nominal values
    labels = data["Type"]
    data = pd.get_dummies(data.iloc[:,1:data.columns.size - 1])
    print(data.columns.size)
    data["Type"] = labels
    data = NanCleaner(data)
    #data = NanRemover(data)
    fs = FeatureSelector(data)
    features = fs.featureSelectionByLogisticRegression(5)
    print(features)
    clfs = [MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42),
            svm.SVC(kernel='rbf', C=1e3, gamma=0.1),
            svm.SVC(kernel='poly', C=1e3, degree=3),
            svm.SVC(kernel = 'linear', C = 1e3),
            svm.SVC()]
    tester = kFolderTester(1, clfs, data, features, "Type")
    tester.startMultiClassificationTest()

