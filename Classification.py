import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Error import Mse
from FeatureSelector import FeatureSelector
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
    #clfs = [MLPClassifier(solver='adam', alpha=10, hidden_layer_sizes=(150,), random_state=1, activation="tanh")]
    clfs = [MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42),tree.DecisionTreeClassifier(),
            svm.SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True),
            svm.SVC(kernel='poly', C=1.0, degree=3, probability=True),
            svm.SVC(kernel = 'linear', C = 1.0, probability=True),
            KNeighborsClassifier()]
    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]
    #clfs = [MLPClassifier(solver='adam', alpha=10, hidden_layer_sizes=(150,), random_state=1, activation="tanh")]

    #clfNames = ["adam"]

    #RandomForestClassifier(n_jobs=10, random_state=45), svm.SVC()
    tester = kFolderTester(1, clfs, data, features, 'shares', clfNames)
    tester.startClassificationTest()

