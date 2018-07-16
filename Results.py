class Results:
    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.k_cohen = 0
        self.f1_measure = 0
        self.log_loss = 0

    def setAttr(self, key, value):
        super.__setattr__(self, key, value)

    def getAttr(self, key, value):
        return super.__getattribute__(self, key)


