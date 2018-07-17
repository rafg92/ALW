from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from DataSplitter import DataSplitter
from FeatureSelector import FeatureSelector
from Results import Results
from kFolderTester import kFolderTester


def printResults(mse, clfName, n_features):
    path = "Regression/Test/" + clfName
    file = Path(path)
    if not file.is_file():
        open(path, "w+")
    f = open(path, "a")
    line = str(n_features) + "," + str(mse) + '\n'
    print(line)
    f.write(line)
    f.close()


if(__name__ == "__main__"):
    np.random.seed(15)

    # Read in data and display first 5 rows
    data = pd.read_csv('training_AMPL.csv', sep=",")

    print('The shape of our data is:', data.shape)

    #one hot encoding: transorming nominal values
    labelName = "G3"
    labels = data[labelName]
    data = pd.get_dummies(data)
    print(data.columns.size)
    print(data)
    data[labelName] = labels

    train, test = DataSplitter().splitData(data.copy())
    Y_train = train[labelName]
    print(train.columns)
    X_train_origin = train.copy().drop(labelName, axis=1)
    print(X_train_origin.columns)
    Y_test = test[labelName]
    X_test_origin = test.copy().drop(labelName, axis=1)

    fs = FeatureSelector(train.copy())
    fsSize = train.columns.size
    threshold = 10

    fs = FeatureSelector(train.copy())
    while(fsSize >= threshold):
        features = fs.featureSelectionRegression(fsSize)
        print("FEATURES NEL WHILE ", features)
        #C=1e3
        svr_rbf = SVR(kernel='rbf', C=1.0, gamma=0.1)
        svr_lin = SVR(kernel='linear', C=1.0)
        svr_poly = SVR(kernel='poly', C=1.0, degree=3)
        clfs = [MLPRegressor(solver='lbfgs', alpha = 10.0, hidden_layer_sizes=(100,), random_state=1, activation="tanh",epsilon=1e-8, early_stopping=True),
        #     RandomForestRegressor(n_jobs=10, random_state=45, n_estimators=5),  DecisionTreeRegressor(), svr_lin, svr_poly, svr_rbf]
        # clfNames = ["lbfgs", "randomForest", "decisionTree", "linear", "poly",  "rbf",]
        ]
        clfNames = ['lbfgs']
        tester = kFolderTester(4, clfs, train.copy(), features, labelName, clfNames)
        tester.startRegressionTest()
        # starting on the evaluation set
        X_train = X_train_origin[features]
        X_test = X_test_origin[features]

        #scaler = StandardScaler()

        #scaler.fit(X_train)
        #X_train = scaler.transform(X_train)
        # apply same transformation to test data
        #X_test = scaler.transform(X_test)
        print(X_train)
        print(Y_train)
        i = 0
        while (i < len(clfs)):
            clfs[i].fit(X_train, Y_train)
            preds = clfs[i].predict(X_test)


            printResults(mean_squared_error(Y_test, preds), clfNames[i], len(features))
            i += 1

        fsSize -= 5