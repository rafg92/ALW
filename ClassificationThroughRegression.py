from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics, tree, preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ClassifficationByRegressionTester import ClassificationByRegressionTester
from DataSplitter import DataSplitter
from Error import Mse
from FeatureSelector import FeatureSelector
from ResultPlotter import Plotter
from Results import Results
from kFolderTester import kFolderTester


def transRow(x):
    # @param x is a row of the dataframe
    if(x[x.size - 1] > 1400):
        x[x.size - 1] = 1
    else:
        x[x.size - 1] = 0
    return x

validRowsIndexes = []
currentIndex = 0

def dataCleaningApply(y):
    # @param y is a row of the dataframe
    global validRowsIndexes
    global currentIndex
    for i in range(0, y.size) :
        if(y[i] < 0):
            return
    validRowsIndexes.append(currentIndex)
    currentIndex = currentIndex + 1
    return


def dataCleaning(x):
    # @param x is a dataframe
    x.apply(dataCleaningApply, axis = 1)

def eliminateWeekApply(x):
    #@param x is a coloumn name
    if("week" in x):
        return False
    return True

def eliminateWeekSections(x):
    #@param x is a list of coloumns' names
    mask = []
    i = 0
    while(i < x.size):
        mask.append(eliminateWeekApply(x[i]))
        i += 1
    return mask

def transLabel(dataframe):

    for i in dataframe.iloc[2:]:
        print (i)
        transRow(i)


def printRegressionResults(mse, clfName, n_features):
    path = "ClassificationByRegression/Test/Regression/"  + clfName
    file = Path(path)
    if not file.is_file():
        open(path, "w+")
    f = open(path, "a")
    line = str(n_features) + "," + str(mse) + '\n'
    print(line)
    f.write(line)
    f.close()

def printClassificationResults(results, clfName, n_features):
    path = "ClassificationByRegression/Test/Classification" + clfName
    file = Path(path)
    if not file.is_file():
        open(path, "w+")
    f = open(path, "a")
    # line = str(len(self.features)) + "," + str(self.results[index].accuracy/self.k) +  "," \
    #        + str(self.results[index].k_cohen / self.k)\
    #        + "," + str(self.results[index].log_loss / self.k) + '\n'
    line = str(n_features) + "," + str(results.accuracy) + "," \
           + str(results.precision) + "," + str(results.recall) \
           + "," + str(results.k_cohen) + "," + str(results.f1_measure) \
           + '\n'
    print(line)
    f.write(line)
    f.close()



if(__name__ == "__main__"):
    # Set random seed
    np.random.seed(12345)

    # Read in data and display first 5 rows
    data = pd.read_csv('classification.csv', sep=";")
    # print("data.head: ")
    # print( data.head())

    print('The shape of our data is:', data.shape)

    # Descriptive statistics for each column
    # print(data.describe())

    data = data.iloc[:, 1:]

    # print(data)

    # One-hot encode the data using pandas get_dummies
    data = pd.get_dummies(data)

    notWeek = eliminateWeekSections(data.columns)
    print(notWeek)
    data = data[data.columns[notWeek]]
    print(data.columns)

    labelName = "shares"

    train, test = DataSplitter().splitData(data.copy())

    print("Splitted")

    print("Fitting")

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(train)

    print("Fitted")

    trainTmp = pd.DataFrame(scaler.transform(train.copy()), columns=train.columns)
    # apply same transformation to test data
    testTmp = pd.DataFrame(scaler.transform(test.copy()), columns=test.columns)
    print("Transformed")

    fsSize = train.columns.size
    threshold = 10
    fs = FeatureSelector(train.copy())

    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]

    while (fsSize >= threshold):
        print("selecting Features")
        features = fs.featureSelectionSelectKBestRegression(fsSize, labelName)
        print("FEATURES NEL WHILE ", features)
        # C=1e3
        svr_rbf = SVR(kernel='rbf', C=1)
        svr_lin = SVR(kernel='linear', C=1)
        svr_poly = SVR(kernel='poly', C=1)

        clfs = [MLPRegressor(solver='lbfgs', alpha=.0001, hidden_layer_sizes=(100,), activation="tanh", epsilon=1e-4),
                MLPRegressor(solver='adam', alpha=.0001, hidden_layer_sizes=(100,), activation="tanh", epsilon=1e-4),
                MLPRegressor(solver='sgd', alpha=.0001, hidden_layer_sizes=(100,), activation="tanh", epsilon=1e-4),
                RandomForestRegressor(n_jobs=10, random_state=45, n_estimators=10), DecisionTreeRegressor(), svr_lin,
                svr_poly, svr_rbf]
        clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "linear", "poly", "rbf", ]

        tester = ClassificationByRegressionTester(1, clfs, train.copy(), features, labelName, clfNames)
        tester.startClassificationByRegressionTest()

        # starting on the evaluation set

        Y_train = trainTmp[labelName]
        X_train = trainTmp[features]
        Y_test = testTmp[labelName]
        X_test = testTmp[features]

        i = 0

        result = Results()
        while (i < len(clfs)):
            print(clfNames[i])
            #taking the result of the regression problem on the test set
            clfs[i].fit(X_train, Y_train)
            preds = clfs[i].predict(X_test)
            printRegressionResults(mean_squared_error(Y_test, preds), clfNames[i], len(features))

            #reconstructing the datafreames to undo the scaling in order to perform labeling transformation

            classPreds = test.copy()
            classPreds[labelName] = preds

            classTest = pd.DataFrame(scaler.inverse_transform(test.copy()), columns=test.columns)
            classPreds = pd.DataFrame(scaler.inverse_transform(classPreds.copy()), columns=classPreds.columns)
            y_test = classTest[labelName]
            preds = classPreds[labelName]

            #tranforming the regression results to classify
            j = 0
            while (j < len(y_test)):
                if (y_test[j] > 1400):
                    y_test[j] = 1
                else:
                    y_test[j] = 0

                if (preds[j] > 1400):
                    preds[j] = 1
                else:
                    preds[j] = 0
                j += 1
            #taking the scores of classification
            y_test = pd.factorize(y_test)[0]
            preds = pd.factorize(preds)[0]
            result.accuracy = metrics.accuracy_score(y_test, preds)
            result.precision = metrics.precision_score(y_test, preds)
            result.recall = metrics.recall_score(y_test, preds)
            result.k_cohen = metrics.cohen_kappa_score(y_test, preds)
            result.f1_measure = metrics.f1_score(y_test, preds)
            printClassificationResults(result, clfNames[i], len(features))

            i += 1

        fsSize -= 5

    dirPath = "ClassificationByRegression/Test/Regression/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Mse"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

    dirPath = "ClassificationByRegression/Train/Regression/"
    plotter = Plotter(clfNames, dirPath)
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

    dirPath = "ClassificationByRegression/Test/Classification"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

    dirPath = "ClassificationByRegression/Train/Classification"
    plotter = Plotter(clfNames, dirPath)
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

