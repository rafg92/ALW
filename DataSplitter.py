import numpy as np


class DataSplitter:
    def __init__(self):
        self.data = None

    def splitData(self, data):
        self.data = data
        self.data['is_train'] = np.random.uniform(0, 1, len(self.data)) <= .75

        # Create two new dataframes, one with the training rows, one with the test rows
        train, test = self.data[self.data['is_train'] == True], self.data[self.data['is_train'] == False]
        self.data = self.data.iloc[:, 0:self.data.columns.size - 1]
        train = train.iloc[:, 0:self.data.columns.size]
        test = test.iloc[:, 0:self.data.columns.size]
        return train, test
