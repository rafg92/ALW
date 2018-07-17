from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
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



class kFolderTester:

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

    def testClassifier(self, clf, train, test, resultIndex):

        #it tests the classifier and updates the relative result component

        # Show the number of observations for the test and training dataframes
        print('Number of observations in the training data:', len(train))
        print('Number of observations in the test data:', len(test))

        # train['species'] contains the actual species names. Before we can use it,
        # we need to convert each species name into a digit. So, in this case there
        # are three species, which have been coded as 0, 1, or 2.

        y = pd.factorize(train[self.labelCol])[0]
        y_test = pd.factorize(test[self.labelCol])[0]

        # Create a random forest Classifier. By convention, clf means 'Classifier'
        # clf = RandomForestClassifier(n_jobs=10, random_state=45)
        # clf.set_params(n_estimators=10)

        #clf = svm.SVC()
        # clf.fit(X, y)

        # Train the Classifier to take the training features and learn how they relate
        # to the training y (the species)
        train = train[self.features]
        test = test[self.features]
        scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(train)
        train = scaler.transform(train)
        # apply same transformation to test data
        test = scaler.transform(test)
        clf.fit(train, y)

        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        preds = clf.predict(test)

        # View the predicted probabilities of the first 10 observations
        # print clf.predict_proba(test[features])[0:10]
        print("		test labels")
        # print(test["shares"][7:15])

        # View the PREDICTED species for the first five observations
        print("		predicted labels")
        # print(clf.predict(test[features])[7:15])

        # View the ACTUAL species for the first five observations
        # print pd.factorize(test['shares'][0]).head()
       # y_test = pd.factorize(test[self.labelCol])[0]

        # Create confusion matrix
        print("\n\n\n\n\n\n TEST: ")
        # print(features_size)
        print(pd.crosstab(y_test, preds, rownames=['Actual Species'], colnames=['Predicted Species']))
        res = self.results[resultIndex]
        res.accuracy += metrics.accuracy_score(y_test, preds)
        res.precision += metrics.precision_score(y_test, preds)
        res.recall += metrics.recall_score(y_test, preds)
        res.k_cohen += metrics.cohen_kappa_score(y_test, preds)
        res.f1_measure += metrics.f1_score(y_test, preds)
        res.log_loss += metrics.log_loss(y_test, clf.predict_proba(test))
        self.results[resultIndex] = res


    def testMultiClassifier(self, clf, train, test, resultIndex):

        #it tests the classifier and updates the relative result component

        # Show the number of observations for the test and training dataframes
        print('Number of observations in the training data:', len(train))
        print('Number of observations in the test data:', len(test))

        # train['species'] contains the actual species names. Before we can use it,
        # we need to convert each species name into a digit. So, in this case there
        # are three species, which have been coded as 0, 1, or 2.
        y = pd.factorize(train[self.labelCol])[0]
        y_test = pd.factorize(test[self.labelCol])[0]

        definitions = pd.factorize(self.data[self.labelCol])[1]

        # Train the Classifier to take the training features and learn how they relate
        # to the training y (the species)
        train = train[self.features]
        test = test[self.features]
        scaler = StandardScaler()

        # Don't cheat - fit only on training data
        scaler.fit(train)
        train = scaler.transform(train)
        # apply same transformation to test data
        test = scaler.transform(test)
        clf.fit(train, y)

        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        preds = clf.predict(test)

        # Create confusion matrix
        #print("\n\n\n\n\n\n TEST: ")
        # print(features_size)
        # Reverse factorize
        reversefactor = dict(zip(range(definitions.size), definitions))
        y1_test = np.vectorize(reversefactor.get)(y_test)
        preds1 = np.vectorize(reversefactor.get)(preds)
        #print(y_test)
        #print("\n\n", preds)
        # Making the Confusion Matrix
        #print(pd.crosstab(y1_test, preds1, rownames=['Actual Species'], colnames=['Predicted Species']))
        res = self.results[resultIndex]
        res.accuracy += metrics.accuracy_score(y_test, preds)
        res.precision += metrics.precision_score(y_test, preds, average="macro")
        res.recall += metrics.recall_score(y_test, preds, average="macro")
        res.k_cohen += metrics.cohen_kappa_score(y_test, preds)
        res.f1_measure += metrics.f1_score(y_test, preds, average="macro")
        res.log_loss += metrics.log_loss(y_test, clf.predict_proba(test))
        self.results[resultIndex] = res


    def testRegressor(self, clf, train, test, resultIndex):

        # Show the number of observations for the test and training dataframes
        print('Number of observations in the training data:', len(train))
        print('Number of observations in the test data:', len(test))

        scaler = StandardScaler()
        # Don't cheat - fit only on training data
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



    def kFoldRegressionTest(self):
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
                self.testRegressor(clf, train, test, i)
                i += 1

    def kFoldClassificationTest(self):
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
                self.testClassifier(clf, train, test, i)
                i += 1

    def kFoldMultiClassificationTest(self):
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
                self.testMultiClassifier(clf, train, test, i)
                i += 1

    def splitClassificationTest(self):

        #simple training with division training/test of .75/.25

        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size]
        test = test.iloc[:, 0:self.data.columns.size]
        i = 0
        for clf in self.classifiers:
            self.testClassifier(clf, train, test, i)
            i += 1

    def splitMultiClassificationTest(self):

        #simple training with division training/test of .75/.25

        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size]
        test = test.iloc[:, 0:self.data.columns.size]
        i = 0
        print("DATA IN TESTER", self.data)
        for clf in self.classifiers:
            self.testMultiClassifier(clf, train, test, i)
            i += 1

    def splitRegressionTest(self):

        #simple training with division training/test of .75/.25

        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size ]
        test = test.iloc[:, 0:self.data.columns.size]
        i = 0
        for clf in self.classifiers:
            self.testRegressor(clf, train, test, i)
            i += 1

    def startRegressionTest(self):
        if (self.k == 1):
            self.splitRegressionTest()
        else:
            self.kFoldRegressionTest()
        print("Result")
        for r in range(0, len(self.mse)):
            self.printResultsRegression(r)

    def startClassificationTest(self):
        if(self.k == 1):
            self.splitClassificationTest()
        else:
            self.kFoldClassificationTest()

        for r in range(0, len(self.results)):
            self.printResultsClassification(r)

    def startMultiClassificationTest(self):
        if(self.k == 1):
            self.splitMultiClassificationTest()
        else:
            self.kFoldMultiClassificationTest()

        for r in range(0, len(self.results)):
            self.printResultsMultiClassification(r)

    def printResultsClassification(self, index):
        path = "Classification/Train/" + self.clfNames[index]
        file = Path(path)
        if not file.is_file():
            open(path, "w+")
        f = open(path, "a")
        line = str(len(self.features)) + "," + str(self.results[index].accuracy/self.k) +  "," \
               + str(self.results[index].precision/self.k) + "," + str(self.results[index].recall/self.k) \
               + "," + str(self.results[index].k_cohen / self.k) + "," + str(self.results[index].f1_measure / self.k)\
               + "," + str(self.results[index].log_loss / self.k) + '\n'
        print(line)
        f.write(line)
        f.close()

    def printResultsMultiClassification(self, index):
        path = "MultiClassification/Train/" + self.clfNames[index]
        file = Path(path)
        if not file.is_file():
            open(path, "w+")
        f = open(path, "a")
        # line = str(len(self.features)) + "," + str(self.results[index].accuracy/self.k) +  "," \
        #        + str(self.results[index].k_cohen / self.k)\
        #        + "," + str(self.results[index].log_loss / self.k) + '\n'
        line = str(len(self.features)) + "," + str(self.results[index].accuracy / self.k) + "," \
               + str(self.results[index].precision / self.k) + "," + str(self.results[index].recall / self.k) \
               + "," + str(self.results[index].k_cohen / self.k) + "," + str(self.results[index].f1_measure / self.k) \
               + "," + str(self.results[index].log_loss / self.k) + '\n'
        print(line)
        f.write(line)
        f.close()


    def printResultsRegression(self, index):
        path = "Regression/Train/" + self.clfNames[index]
        file = Path(path)
        if not file.is_file():
            open(path, "w+")
        f = open(path, "a")
        line = str(len(self.features)) + "," + str(self.mse[index]/self.k) + '\n'
        print(line)
        f.write(line)
        f.close()



