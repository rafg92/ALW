import numpy as np
import pandas as pd


class DataSplitter:
    def __init__(self):
        self.data = None

    def splitData(self, data, percent = .75):
        self.data = data
        # self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= percent
        #
        # # Create two new dataframes, one with the training rows, one with the test rows
        # train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        # self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        # train = train.iloc[:, 0:self.data.columns.size]
        # test = test.iloc[:, 0:self.data.columns.size]
        self.data = self.data.reindex(np.random.permutation(self.data.index))
        lenght = int(self.data.index.size * percent)
        train = self.data.iloc[0:lenght, :]
        test = self.data.iloc[lenght: self.data.index.size, :]
        return train, test

    def splitDataEqually(self, data, labelName, percent = .75):
        my_data = data.copy()

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



