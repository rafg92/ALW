
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE


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