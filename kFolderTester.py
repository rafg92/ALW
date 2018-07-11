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




class kFolderTester:

    def __init__(self, k, classifiers, data, features, label):
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

    def createFolders(self):
        #it prepares the folders for k-fold cross validation

        folders = [[]] * self.k
        dataTmp = self.data
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
        self.results[resultIndex] = res


    def testRegressor(self, clf, train, test, resultIndex):

        # Show the number of observations for the test and training dataframes
        print('Number of observations in the training data:', len(train))
        print('Number of observations in the test data:', len(test))

        y = train[self.labelCol]
        y_test = test[self.labelCol]

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

    def splitClassificationTest(self):

        #simple training with division training/test of .75/.25

        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size - 1]
        test = test.iloc[:, 0:self.data.columns.size - 1]
        i = 0
        for clf in self.classifiers:
            self.testClassifier(clf, train, test, i)
            i += 1

    def splitRegressionTest(self):

        #simple training with division training/test of .75/.25

        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size - 1]
        test = test.iloc[:, 0:self.data.columns.size - 1]
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
        for r in self.mse:
            print(r/self.k)
            print("\n\n")


    def startClassificationTest(self):
        if(self.k == 1):
            self.splitClassificationTest()
        else:
            self.kFoldClassificationTest()

        for r in self.results:
            print(r.accuracy / self.k)
            print(r.precision / self.k)
            print(r.recall / self.k)
            print(r.k_cohen / self.k)
            print(r.f1_measure / self.k)
            print("\n\n")






if(__name__ == "__main__"):
    # Set random seed
    np.random.seed(105)

    # Read in data and display first 5 rows
    data = pd.read_csv('training_R.csv', sep=";")
    # print("data.head: ")
    # print( data.head())

    print('The shape of our data is:', data.shape)

    # Descriptive statistics for each column
    # print(data.describe())

    data = data.iloc[:, 1:]

    # print(data)

    # One-hot encode the data using pandas get_dummies
    data = pd.get_dummies(data)

    data = data.apply(transRow, axis=1)

    notWeek = eliminateWeekSections(data.columns)
    print(notWeek)
    data = data[data.columns[notWeek]]
    print(data.columns)
    fs = FeatureSelector(data)
    features = fs.featureSelectionByLogisticRegression(40)
    print(features)
    clfs = [MLPClassifier(solver='adam', alpha=10, hidden_layer_sizes=(150,150,), random_state=1, activation="tanh")]
    #RandomForestClassifier(n_jobs=10, random_state=45), svm.SVC()
    tester = kFolderTester(3, clfs, data, features, 'shares')
    tester.startClassificationTest()

