import numpy as np
import pandas as pd
from sklearn import metrics, tree
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
    data = pd.get_dummies(data.iloc[:,1:data.columns.size - 1])
    print(data.columns.size)
    data[labelName] = labels
    #data = NanCleaner(data)
    #data = NanRemover(data)

    train, test = DataSplitter().splitData(data)
    Y_train = train[labelName]
    X_train = train.iloc[:, 0:train.columns.size - 1]
    Y_test = test[labelName]
    X_test = test.iloc[:, 0:test.columns.size - 1]

    fs = FeatureSelector(train)
    features = fs.featureSelectionByLogisticRegression(train.columns.size)
    print(features)
    clfs = [MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            MLPClassifier(solver='adam', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            MLPClassifier(solver='sgd', alpha=0.1, hidden_layer_sizes=(150,), random_state=1, activation="tanh", max_iter=500),
            RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42), tree.DecisionTreeClassifier(),
            svm.SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True),
            svm.SVC(kernel='poly', C=1.0, degree=3, probability=True),
            svm.SVC(kernel = 'linear', C = 1.0, probability=True),
            KNeighborsClassifier()]
    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]
    tester = kFolderTester(1, clfs, train, features, labelName, clfNames)
    tester.startMultiClassificationTest()


    #starting on the evaluation set
    X_train = X_train[features]
    Y_train = Y_train[features]

    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    clfs[1].fit(X_train, Y_train)
    preds = clfs[1].predict(X_test)

    print(metrics.accuracy_score(Y_test, preds))
    print(metrics.precision_score(Y_test, preds, average="macro"))
    print(metrics.recall_score(Y_test, preds, average="macro"))
    print(metrics.cohen_kappa_score(Y_test, preds))
    print(metrics.f1_score(Y_test, preds, average="macro"))
    print(metrics.log_loss(Y_test, clfs[1].predict_proba(test)))


