
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

#used to make plots of the results
class Plotter:

    def __init__(self,names, dirPaths):
        self.dirPaths = dirPaths
        self.names = names
        self.dataframes = []
        for i in range(0, len(self.names)):
            df = pd.read_csv(self.dirPaths + self.names[i], sep=',', header=None)
            self.dataframes.append(df)



    def plotMetric(self, metricName, index):
        c = 0
        for df in self.dataframes:
            print(df)

            plt.plot(df[0].values, df[index].values,  color = "C"+str(c), label = self.names[c])
            plt.legend(loc = "best")
            c += 1
        plt.savefig(metricName)
        plt.show()


#convenience main
if(__name__ == "__main__"):
    #clfsPaths = ["Regression/Test/lbfgs","Regression/Test/linear","Regression/Test/rbf"]
    # clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "linear", "poly", "rbf", ]
    # dirPath = "Regression/Train/"
    # plotter = Plotter(clfNames, dirPath)
    # plotter.plotMetric("Accuracy", 1)
    #clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]
    #clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "linear", "knn"]
    clfNames = ["knn", "knn10", "knn20"]

    dirPath = "MultiClassification/Test/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    # metricNames = ["Mse"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1

    dirPath = "MultiClassification/Train/"
    plotter = Plotter(clfNames, dirPath)
    # metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i + 1)
        i += 1



