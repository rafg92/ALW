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

if(__name__ == "__main__"):
    np.random.seed(15)

    # Read in data and display first 5 rows
    data = pd.read_csv('training_AMPL.csv', sep=",")

    print('The shape of our data is:', data.shape)

    #one hot encoding: transorming nominal values
    data = pd.get_dummies(data)
    print(data.columns)
    fs = FeatureSelector(data)
    features = fs.featureSelectionByLogisticRegression(data.columns.size)
    print(features)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=3)
    clfs = [MLPRegressor(solver='lbfgs', alpha=10, hidden_layer_sizes=(100,), random_state=1, activation="tanh",epsilon=1e-4),
    RandomForestRegressor(n_jobs=10, random_state=45), svr_lin, svr_poly, svr_rbf]
    tester = kFolderTester(1, clfs, data, features, "G3")
    tester.startRegressionTest()
