import numpy as np
import pandas as pd


class DataSplitter:
    def __init__(self):
        self.data = None

    def splitData(self, data, percent = .75):
        # Create two new dataframes, one with the training rows, one with the test rows
        #@param data: it's a dataframe
        self.data = data

        self.data = self.data.reindex(np.random.permutation(self.data.index))
        lenght = int(self.data.index.size * percent)
        train = self.data.iloc[0:lenght, :]
        test = self.data.iloc[lenght: self.data.index.size, :]
        return train, test

    def splitDataEqually(self, data, labelName, percent = .75):
        my_data = data.copy()

        #grouping the examples by their label
        grouped = my_data.groupby(labelName, as_index = False)

        # initializing empty data frames
        training_set = pd.DataFrame(columns = my_data.columns)
        test_set = pd.DataFrame(columns = my_data.columns)

        # binding all the splitting subsets to make a whole
        for group in grouped:
            train, test = self.splitData( group[1], percent)
            training_set = training_set.append(train, ignore_index = True)
            test_set = test_set.append(test, ignore_index=True)

        print(training_set[labelName].unique())
        print(test_set[labelName].unique())

        return training_set, test_set



