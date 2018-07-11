class Mse:

    def __init__(self):
        self=self

    def calculate(self, actual, pred):

        assert len(actual) == len(pred)
        mse = 0
        n_examples = len(actual)
        for i in range(0, n_examples):
            mse += (actual[i] - pred[i])**2
        return mse/n_examples