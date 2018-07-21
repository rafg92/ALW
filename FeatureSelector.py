
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, f_regression, f_classif


class FeatureSelector:
    def __init__(self, data):
        self.data = data

    def featureSelectionByLogisticRegression(self, featureSize):
        array = self.data.values
        X = array[:, 0:self.data.columns.size - 1]
        Y = pd.factorize(array[:, self.data.columns.size - 1])[0]
        # feature extraction
        model = LogisticRegression()
        rfe = RFE(model, featureSize)
        fit = rfe.fit(X, Y)
        print("Num Features: ")
        print(fit.n_features_)
        print("Selected Features: ")
        print(fit.support_)
        # print("Feature Ranking: %s") % fit.ranking_

        features = (self.data.columns[0:self.data.columns.size - 1])[fit.support_]
        return features

    def featureSelectionRegression(self, featureSize, labelName):
        #feature selection for regression tasks
        data = self.data.copy()
        labels = data[labelName]
        train = data.drop(labelName, axis=1)

        X = train.values
        Y = labels.values
        # feature extraction
        model = LogisticRegression()
        rfe = RFE(model, featureSize)
        fit = rfe.fit(X, Y)
        print("Num Features: ")
        print(fit.n_features_)
        print("Selected Features: ")
        print(fit.support_)
        # print("Feature Ranking: %s") % fit.ranking_

        features = (train.columns)[fit.support_]
        return features

    def featureSelectionSelectKBestRegression(self, featureSize, labelName):
        data = self.data.copy()
        labels = data[labelName]
        train = data.drop(labelName, axis=1)
        if(featureSize > train.columns.size):
            featureSize = 'all'
        X = train.values
        Y = labels.values
        # feature extraction
        fit = SelectKBest(f_regression, k = featureSize).fit(X,Y)
        # print("Num Features: ")
        # print(fit.n_features_)
        # print("Selected Features: ")
        # print(fit.support_)
        # print("Feature Ranking: %s") % fit.ranking_

        features = train.columns[fit.get_support()]
        return features

    def featureSelectionSelectKBestClassification(self, featureSize, labelName):
        data = self.data.copy()
        labels = data[labelName]
        train = data.drop(labelName, axis=1)
        if (featureSize > train.columns.size):
            featureSize = 'all'
        X = train.values
        Y = labels.values
        # feature extraction
        fit = SelectKBest(f_regression, k = featureSize).fit(X,Y)

        # print("Num Features: ")
        # print(fit.n_features_)
        # print("Selected Features: ")
        # print(fit.support_)
        # print("Feature Ranking: %s") % fit.ranking_
        features = train.columns[fit.get_support()]

        return features