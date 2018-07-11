from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

#Load scipy
import scipy as sc

# Load feature selection
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from functools import partial


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

def test(data, features_size):

    array = data.values
    X = array[:, 0:data.columns.size - 1]
    Y = pd.factorize(array[:, data.columns.size - 1])[0]
    # feature extraction
    model = LogisticRegression()
    rfe = RFE(model, features_size)
    fit = rfe.fit(X, Y)
    print("Num Features: ")
    print(fit.n_features_)
    print("Selected Features: ")
    print(fit.support_)
    # print("Feature Ranking: %s") % fit.ranking_

    # View features
    # dummie = [False False False False  True  True False False False False False False
    #  True  True  True  True False  True False False False False False False
    # False False False False False False False  True  True  True False  True
    #  True  True  True  True  True  True  True  True  True  True  True False
    #  True False  True False  True  True  True  True  True  True False]
    features = (data.columns[0:data.columns.size - 1])[fit.support_]
    print(features)

    k = 10

    folders = [[]]*k
    dataTmp = data
    for u in range(0,k):
        # Create a new column that for each row, generates a random number between 0 and 1, and
        # if that value is less than or equal to .75, then sets the value of that cell as True
        # and false otherwise. This is a quick and dirty way of randomly assigning some rows to
        # be used as the training data and some as the test data.

        dataTmp['is_train'] = np.random.uniform(0, 1, len(dataTmp)) <= 1.0/(k-u)
        folder, dataTmp = dataTmp[dataTmp['is_train'] == True], dataTmp[dataTmp['is_train'] == False]
        print(dataTmp.shape)
        folders[u] = folder.iloc[:, 0:folder.columns.size - 1]
        # folders[u] = dataTmp[booleanMask]
        # dataTmp = dataTmp[booleanMask==False]

    accuracy = 0
    precision = 0
    recall = 0
    k_cohen = 0
    f1_measure = 0
    for u in range(0,k):
        train = pd.DataFrame()
        for j in range(0,k):
            if(j!=u):
                train = train.append(folders[j])

        test = folders[u]
    #train, test = data[data['is_train'] == True], data[data['is_train'] == False]

        train = pd.DataFrame(train, columns=data.columns)
        test = pd.DataFrame(test, columns= data.columns)


        # Show the number of observations for the test and training dataframes
        print('Number of observations in the training data:', len(train))
        print('Number of observations in the test data:', len(test))


        # train['species'] contains the actual species names. Before we can use it,
        # we need to convert each species name into a digit. So, in this case there
        # are three species, which have been coded as 0, 1, or 2.
        y = pd.factorize(train['shares'])[0]

        # View target
        #print(y[7:15])

        # Create a random forest Classifier. By convention, clf means 'Classifier'
        #clf = RandomForestClassifier(n_jobs=10, random_state=45)
        #clf.set_params(n_estimators=10)

        clf = svm.SVC()
        #clf.fit(X, y)

        # Train the Classifier to take the training features and learn how they relate
        # to the training y (the species)
        clf.fit(train[features], y)

        # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
        preds = clf.predict(test[features])

        # View the predicted probabilities of the first 10 observations
        # print clf.predict_proba(test[features])[0:10]
        print("		test labels")
        #print(test["shares"][7:15])

        # View the PREDICTED species for the first five observations
        print("		predicted labels")
        #print(clf.predict(test[features])[7:15])

        # View the ACTUAL species for the first five observations
        # print pd.factorize(test['shares'][0]).head()
        y_test = pd.factorize(test["shares"])[0]
        # Create confusion matrix
        print("\n\n\n\n\n\n TEST: ")
        #print(features_size)
        print(pd.crosstab(y_test, preds, rownames=['Actual Species'], colnames=['Predicted Species']))
        accuracy += metrics.accuracy_score(y_test, preds)
        precision += metrics.precision_score(y_test, preds)
        recall += metrics.recall_score(y_test, preds)
        k_cohen += metrics.cohen_kappa_score(y_test, preds)
        f1_measure += metrics.f1_score(y_test, preds)
        print("\n\n\n\n")

        # View a list of the features and their importance scores
        # print list(zip(train[features], clf.feature_importances_))

    print(accuracy/k)
    print(precision/k)
    print(recall/k)
    print(k_cohen/k)
    print(f1_measure/k)



def classifierWork():

    # Set random seed
    np.random.seed(0)


    # Read in data and display first 5 rows
    data = pd.read_csv('training_R.csv', sep = ";")
    #print("data.head: ")
    #print( data.head())

    print('The shape of our data is:', data.shape)

    # Descriptive statistics for each column
    #print(data.describe())

    data = data.iloc[:, 1:]

    #print(data)

    # One-hot encode the data using pandas get_dummies
    data = pd.get_dummies(data)

    # Display the first 5 rows of the last 12 columns
    #print(data.iloc[: ,5:].head(5))

    # DATA CLEANING
    # validRowsIndexes = dataCleaning(data)
    #dataCleaning(data)
    # print validRowsIndexes
    #data = data.iloc[validRowsIndexes, :]
    #print(data)
    # transformation labeling .......

    data = data.apply(transRow, axis = 1)

    notWeek = eliminateWeekSections(data.columns)
    print(notWeek)
    data = data[data.columns[notWeek]]
    print(data.columns)

    #print("STANPA DOPO TRASFORMATION")
    #print(data)
    n_features = 10
    while(n_features < data.columns.size):
        test(data, n_features)
        n_features += 10


if (__name__ == "__main__"):

    classifierWork()





    #########################/********************************************************************************************************************
    # edicted Species     0     1
    # Actual Species
    # 0                  1597  1789
    # 1                  2435  1003
    # Actual
    # Species
    # 0
    # 1689
    # 1697
    # 1
    # 2438
    # 1000


