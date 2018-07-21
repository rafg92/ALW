from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Error import Mse
from FeatureSelector import FeatureSelector
from Results import Results



class ClassificationByRegressionTester:

    def __init__(self, k, classifiers, data, features, label, clfNames):
        self.k = k
        self.classifiers = classifiers
        self.results = []
        self.data = data
        self.features = features
        self.labelCol = label
        self.folders = []
        self.mse = [0]*len(self.classifiers)
        for i in range(0, len(self.classifiers)):
            self.results.append(Results())
        self.clfNames = clfNames

    def createFolders(self):
        #it prepares the folders for k-fold cross validation

        folders = [[]] * self.k
        dataTmp = self.data.copy()
        for u in range(0, self.k):
            # Create a new column that for each row, generates a random number between 0 and 1, and
            # if that value is less than or equal to .75, then sets the value of that cell as True
            # and false otherwise. This is a quick and dirty way of randomly assigning some rows to
            # be used as the training data and some as the test data.

            dataTmp['is_train'] = np.random.uniform(0, 1, len(dataTmp)) <= 1.0 / (self.k - u)
            folder, dataTmp = dataTmp[dataTmp['is_train'] == True], dataTmp[dataTmp['is_train'] == False]
            print(dataTmp.shape)
            folders[u] = folder.iloc[:, 0:folder.columns.size - 1]
            # folders[u] = dataTmp[booleanMask]
            # dataTmp = dataTmp[booleanMask==False]
        self.folders = folders


    def testClassificationByRegressor(self, clf, train, test, resultIndex):

        # Show the number of observations for the test and training dataframes
        print('Number of observations in the training data:', len(train))
        print('Number of observations in the test data:', len(test))

        #scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        print(train)
        scaler.fit(train)
        train = pd.DataFrame(scaler.transform(train), columns=train.columns)
        # apply same transformation to test data
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)

        y = train[self.labelCol]
        y_test = test[self.labelCol]

        train = train[self.features]
        test = test[self.features]

        clf.fit(train, y)

        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        preds = clf.predict(test)

        print("\n\n\n\n\n\n TEST: ")
        # print(features_size)
        trError = mean_squared_error(y, clf.predict(train))
        print("ERROR ON TRAINING: ", np.sqrt(trError))
        mse = Mse()
        err = mse.calculate(y_test.values, preds)
        print("OUR ERROR: ", err)
        meanSquaredError = mean_squared_error(y_test, preds)
        print("MSE:", meanSquaredError)

        self.mse[resultIndex] += meanSquaredError

        self.classificationByRegressionResults(test, y_test, preds, scaler, resultIndex)

    def classificationByRegressionResults(self, test, y_test, preds, scaler, resultIndex):
        i = 0
        lenght = len(y_test)
        testTmp = test.copy()
        testTmp[self.labelCol] = y_test
        testTmp = pd.DataFrame(scaler.inverse_transform(testTmp), columns=testTmp.columns)
        y_test = testTmp[self.labelCol]
        testTmp = test.copy()
        testTmp[self.labelCol] = preds
        testTmp = pd.DataFrame(scaler.inverse_transform(testTmp), columns=testTmp.columns)
        preds = testTmp[self.labelCol]

        while( i < lenght):
            if(y_test[i] > 1400):
                y_test[i] = 1
            else:
                y_test[i] = 0

            if(preds[i] > 1400):
                preds[i] = 1
            else:
                preds[i] = 0
            i += 1

        y_test = pd.factorize(y_test)[0]
        preds = pd.factorize(preds)[0]

        print(preds)
        print(y_test)

        print(pd.crosstab(y_test, preds, rownames=['Actual Species'], colnames=['Predicted Species']))
        res = self.results[resultIndex]
        res.accuracy += metrics.accuracy_score(y_test, preds)
        res.precision += metrics.precision_score(y_test, preds)
        res.recall += metrics.recall_score(y_test, preds)
        res.k_cohen += metrics.cohen_kappa_score(y_test, preds)
        res.f1_measure += metrics.f1_score(y_test, preds)
        self.results[resultIndex] = res


    def kFoldClassificationByRegressionTest(self):
        self.createFolders()
        for u in range(0, self.k):
            train = pd.DataFrame()
            for j in range(0, self.k):
                if (j != u):
                    train = train.append(self.folders[j])

            test = self.folders[u]
            train = pd.DataFrame(train, columns=self.data.columns)
            test = pd.DataFrame(test, columns=self.data.columns)

            i = 0
            for clf in self.classifiers:
                self.testClassificationByRegressor(clf, train, test, i)
                i += 1


    def splitClassificationByRegressionTest(self):

        #simple training with division training/test of .75/.25

        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size ]
        test = test.iloc[:, 0:self.data.columns.size]
        i = 0
        for clf in self.classifiers:
            self.testClassificationByRegressor(clf, train, test, i)
            i += 1

    def startClassificationByRegressionTest(self):
        if (self.k == 1):
            self.splitClassificationByRegressionTest()
        else:
            self.kFoldClassificationByRegressionTest()
        print("Result")
        for r in range(0, len(self.mse)):
            self.printResultsRegression(r)
            self.printResultsClassification(r)


    def printResultsClassification(self, index):
        path = "ClassificationByRegression/Train/Classification/" + self.clfNames[index]
        file = Path(path)
        if not file.is_file():
            open(path, "w+")
        f = open(path, "a")
        line = str(len(self.features)) + "," + str(self.results[index].accuracy/self.k) +  "," \
               + str(self.results[index].precision/self.k) + "," + str(self.results[index].recall/self.k) \
               + "," + str(self.results[index].k_cohen / self.k) + "," + str(self.results[index].f1_measure / self.k)\
               + '\n'
        print(line)
        f.write(line)
        f.close()


    def printResultsRegression(self, index):
        path = "ClassificationByRegression/Train/Regression/" + self.clfNames[index]
        file = Path(path)
        if not file.is_file():
            open(path, "w+")
        f = open(path, "a")
        line = str(len(self.features)) + "," + str(self.mse[index]/self.k) + '\n'
        print(line)
        f.write(line)
        f.close()



