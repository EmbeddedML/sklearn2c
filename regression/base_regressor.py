class BaseRegressor:
    def __init__(self, regressor) -> None:
        self.reg = regressor
    def train(self, train_samples, train_labels, **kwargs):
        return self.reg.fit(train_samples, train_labels, **kwargs)
    def inference(self, test_samples, test_labels = None):
        if test_labels is None:
            return self.reg.predict(test_samples)
        else:
            return self.reg.score(test_samples, test_labels)