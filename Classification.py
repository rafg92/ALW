from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import tree, preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from DataSplitter import DataSplitter
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

def printResults(results, clfName, n_features):
    path = "Classification/Test/" + clfName
    file = Path(path)
    if not file.is_file():
        open(path, "w+")
    f = open(path, "a")
    line = str(n_features) + "," + str(results.accuracy) + "," \
           + str(results.precision) + "," + str(results.recall) \
           + "," + str(results.k_cohen) + "," + str(results.f1_measure) \
           + "," + str(results.log_loss) + '\n'
    print(line)
    f.write(line)
    f.close()



if(__name__ == "__main__"):
    # Set random seed
    np.random.seed(12345)

    # Read in data
    data = pd.read_csv('classification.csv', sep=";")

    print('The shape of our data is:', data.shape)

    #discarding the urls column
    data = data.iloc[:, 1:]

    # One-hot encode the data using pandas get_dummies
    data = pd.get_dummies(data)

    # data transformation: real values into labels to classify
    data = data.apply(transRow, axis=1)

    notWeek = eliminateWeekSections(data.columns)
    print(notWeek)
    data = data[data.columns[notWeek]]
    print(data.columns)

    labelName = "shares"

    train, test = DataSplitter().splitDataEqually(data, labelName)
    Y_train = pd.factorize(train[labelName])[0]
    X_train_origin = train.iloc[:, 0:train.columns.size - 1].copy()
    Y_test = pd.factorize(test[labelName])[0]
    X_test_origin = test.iloc[:, 0:test.columns.size - 1].copy()

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(X_train_origin)
    #scaling of training data
    X_train_origin = pd.DataFrame(scaler.transform(X_train_origin.copy()), columns=X_train_origin.columns)
    # apply same transformation to test data
    X_test_origin = pd.DataFrame(scaler.transform(X_test_origin.copy()), columns=X_test_origin.columns)

    trainTmp = X_train_origin.copy()
    trainTmp[labelName] = Y_train
    fs = FeatureSelector(trainTmp)

    featureSize = data.columns.size
    threshold = 10

    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]

    while(featureSize >= threshold):
        features = fs.featureSelectionSelectKBestClassification(featureSize,labelName)
        print(features)
        clfs = [MLPClassifier(solver='lbfgs', alpha=10.0, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
                MLPClassifier(solver='adam', alpha=10.0, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
                MLPClassifier(solver='sgd', alpha=10.0, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
                RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42),tree.DecisionTreeClassifier(),
                svm.SVC(kernel='rbf', C=10.0, gamma=0.1, probability=True),
                svm.SVC(kernel='poly', C=10.0, degree=3, probability=True),
                svm.SVC(kernel = 'linear', C = 10.0, probability=True),
                KNeighborsClassifier(n_neighbors=10)]

        tester = kFolderTester(1, clfs, train.copy(), features, labelName, clfNames)

        #testing on the training data through k-cross-fold validation
        tester.startClassificationTest()

        # starting on the evaluation set
        X_train = X_train_origin[features]
        X_test = X_test_origin[features]

        i = 0
        result = Results()

        #getting test results for each classifier
        while (i < len(clfs)):
            clfs[i].fit(X_train, Y_train)
            preds = clfs[i].predict(X_test)

            result.accuracy = metrics.accuracy_score(Y_test, preds)
            result.precision = metrics.precision_score(Y_test, preds)
            result.recall = metrics.recall_score(Y_test, preds)
            result.k_cohen = metrics.cohen_kappa_score(Y_test, preds)
            result.f1_measure = metrics.f1_score(Y_test, preds)
            result.log_loss = metrics.log_loss(Y_test, clfs[i].predict_proba(X_test))
            #write results into file
            printResults(result, clfNames[i], len(features))
            i += 1

        featureSize -= 5

    #plotting test and train results
    dirPath = "Classification/Test/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

    dirPath = "Classification/Train/"
    plotter = Plotter(clfNames, dirPath)
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1



