from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics, tree, preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from DataSplitter import DataSplitter
from FeatureSelector import FeatureSelector
from ResultPlotter import Plotter
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

def printResults(results, clfName, n_features):
    path = "MultiClassification/Test/" + clfName
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
           + "," + str(results.log_loss) + '\n'
    print(line)
    f.write(line)
    f.close()



if(__name__ == "__main__"):
    np.random.seed(12345)
    #fileName = 'glasses.csv'
    fileName = "Frogs_MFCCs.csv"
    # Read in data and display first 5 rows
    data = pd.read_csv(fileName, sep=",")

    print('The shape of our data is:', data.shape)
    #colnames = ["RI","Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type"]
    #one hot encoding: transorming nominal values
    labelName = "Species"
    labels = data[labelName]
    data = pd.get_dummies(data.iloc[:,0:data.columns.size - 1])
    print(data.columns.size)
    data[labelName] = labels
    #data = NanCleaner(data)
    #data = NanRemover(data)

    train, test = DataSplitter().splitData(data.copy())
    Y_train = pd.factorize(train[labelName])[0]
    X_train_origin = train.iloc[:, 0:train.columns.size - 1].copy()
    Y_test = pd.factorize(test[labelName])[0]
    X_test_origin = test.iloc[:, 0:test.columns.size - 1].copy()

    #scaler = StandardScaler()
    # X_train_minmax = min_max_scaler.fit_transform(X_train)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    scaler.fit(X_train_origin)
    X_train_origin = pd.DataFrame(scaler.transform(X_train_origin.copy()), columns= X_train_origin.columns)
    # apply same transformation to test data
    X_test_origin = pd.DataFrame(scaler.transform(X_test_origin.copy()), columns= X_test_origin.columns)

    featureSize = train.columns.size
    threshold = 5

    fs = FeatureSelector(train.copy())

    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]

    while(featureSize >= threshold):
        features = fs.featureSelectionByLogisticRegression(featureSize)
        print(features)
        # clfs = [MLPClassifier(solver='adam', alpha=10, hidden_layer_sizes=(150,), random_state=1, activation="tanh")]
        #
        # clfNames = ["adam"]
        clfs = [MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
                MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
                MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
                RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42), tree.DecisionTreeClassifier(),
                svm.SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True),
                svm.SVC(kernel='poly', C=1.0, degree=3, probability=True),
                svm.SVC(kernel = 'linear', C = 1.0, probability=True),
                KNeighborsClassifier()]
        tester = kFolderTester(10, clfs, train.copy(), features, labelName, clfNames)
        tester.startMultiClassificationTest()


        #starting on the evaluation set
        X_train = X_train_origin[features]
        X_test = X_test_origin[features]


        i = 0
        result = Results()
        while(i < len(clfs)):
            clfs[i].fit(X_train, Y_train)
            preds = clfs[i].predict(X_test)

            result.accuracy = metrics.accuracy_score(Y_test, preds)
            result.precision = metrics.precision_score(Y_test, preds, average="macro")
            result.recall = metrics.recall_score(Y_test, preds, average="macro")
            result.k_cohen = metrics.cohen_kappa_score(Y_test, preds)
            result.f1_measure = metrics.f1_score(Y_test, preds, average="macro")
            result.log_loss = metrics.log_loss(Y_test, clfs[i].predict_proba(X_test))
            printResults(result, clfNames[i], len(features))
            i += 1

        featureSize -= 5
    dirPath = "MultiClassification/Test/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    i = 0
    while( i < len(metricNames)):
        plotter.plotMetric( dirPath + metricNames[i]+".png", i + 1)
        i += 1

    dirPath = "MultiClassification/Train/"
    plotter = Plotter(clfNames, dirPath)
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1
