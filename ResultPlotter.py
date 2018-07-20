# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly.figure_factory as FF

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


class Plotter:

    def __init__(self,names, dirPaths):
        self.dirPaths = dirPaths
        self.names = names
        self.dataframes = []
        for i in range(0, len(self.names)):
            df = pd.read_csv(self.dirPaths + self.names[i], sep=',', header=None)
            self.dataframes.append(df)

        #self.colors = [plt.cmap(i) for i in np.linspace(0,1, len(paths))]

    def plotMetric(self, metricName, index):
        #index = str(index)
        plotter = plt.figure()
        c = 0
        for df in self.dataframes:
            print(df)
            #trace1 = go.Scatter(
            #    x=df[0], y=df[index],  # Data
            #    mode='lines', name=metricName  # Additional options
            #)
            #trace2 = go.Scatter(x=df['x'], y=df['sinx'], mode='lines', name='sinx')
            #trace3 = go.Scatter(x=df['x'], y=df['cosx'], mode='lines', name='cosx')

            #layout = go.Layout(title='Simple Plot from csv data',
            #                   plot_bgcolor='rgb(160, 230, 30)')

            #fig = go.Figure(data=[trace1], layout=layout)

            # Plot data in the notebook
            #py.iplot(fig, filename='simple-plot-from-csv')
            plt.plot(df[0].tail(4).values, df[index].tail(4).values,  color = "C"+str(c), label = self.names[c])
            plt.legend(loc = "best")
            #plt.savefig
            c += 1
        plt.savefig(metricName)
        plt.show()

if(__name__ == "__main__"):
    #clfsPaths = ["Regression/Test/lbfgs","Regression/Test/linear","Regression/Test/rbf"]
    # clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "linear", "poly", "rbf", ]
    # dirPath = "Regression/Train/"
    # plotter = Plotter(clfNames, dirPath)
    # plotter.plotMetric("Accuracy", 1)
    clfNames = ["lbfgs", "adam", "sgd", "randomForest", "decisionTree", "rbf", "poly", "linear", "knn"]

    dirPath = "MultiClassification/Test/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i+1)
        i += 1

    dirPath = "MultiClassification/Train/"
    plotter = Plotter(clfNames, dirPath)
    metricNames = ["Accuracy", "Precision", "Recall", "K_cohen", "F1_measure", "Log-loss"]
    i = 0
    while (i < len(metricNames)):
        plotter.plotMetric(dirPath + metricNames[i] + ".png", i+1)
        i += 1


