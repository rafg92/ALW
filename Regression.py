from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
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
from ResultPlotter import Plotter
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
    np.random.seed(12345)

    # Read in data and display first 5 rows
    data = pd.read_csv('training_AMPL.csv', sep=",")

    print('The shape of our data is:', data.shape)

    #one hot encoding: transorming nominal values
    labelName = "G3"
    labels = data[labelName]
    data = pd.get_dummies(data)
    data[labelName] = labels

    train, test = DataSplitter().splitData(data.copy())

    print(train.copy())

    #preparing test and training for final evaluation: using copies not to create problems
    #scaler = StandardScaler()

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(train)

    trainTmp = pd.DataFrame(scaler.transform(train.copy()), columns=train.columns)
    # apply same transformation to test data
    testTmp = pd.DataFrame(scaler.transform(test.copy()), columns=test.columns)

    fsSize = train.columns.size
    threshold = 10

    fs = FeatureSelector(train.copy())

    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "linear", "poly", "rbf", ]

    while(fsSize >= threshold):
        features = fs.featureSelectionRegression(fsSize, labelName)
        print("FEATURES NEL WHILE ", features)
        #C=1e3
        svr_rbf = SVR(kernel='rbf', C=1)
        svr_lin = SVR(kernel='linear', C=1)
        svr_poly = SVR(kernel='poly', C=1)

        clfs = [MLPRegressor(solver='lbfgs', alpha = 10.0, hidden_layer_sizes=(10,), activation="tanh",epsilon=1e-4),
                MLPRegressor(solver='adam', alpha=10.0, hidden_layer_sizes=(10,), activation="tanh", epsilon=1e-4),
                MLPRegressor(solver='sgd', alpha=10.0, hidden_layer_sizes=(10,), activation="tanh", epsilon=1e-4),
                RandomForestRegressor(n_jobs=10, random_state=45, n_estimators=10),  DecisionTreeRegressor(), svr_lin, svr_poly, svr_rbf]

        tester = kFolderTester(4, clfs, train.copy(), features, labelName, clfNames)
        tester.startRegressionTest()


        # starting on the evaluation set

        Y_train = trainTmp[labelName]
        X_train = trainTmp[features]
        Y_test = testTmp[labelName]
        X_test = testTmp[features]

        i = 0
        while (i < len(clfs)):
            clfs[i].fit(X_train, Y_train)
            preds = clfs[i].predict(X_test)
            printResults(mean_squared_error(Y_test, preds), clfNames[i], len(features))
            i += 1

        fsSize -= 5

    dirPath = "Regression/Test/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Mse"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

    dirPath = "Regression/Train/"
    plotter = Plotter(clfNames, dirPath)
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

