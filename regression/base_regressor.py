import joblib


class BaseRegressor:
    def __init__(self, regressor) -> None:
        self.reg = regressor

    def train(self, train_samples, train_labels, save_path, **kwargs):
        self.reg = self.reg.fit(train_samples, train_labels, **kwargs)
        if save_path:
            joblib.dump(self, save_path)
    
    def load(self, filename):
        with open(filename, "rb") as joblib_file:
            saved_model = joblib.load(joblib_file)
        self.__dict__.update(saved_model.__dict__)

    def inference(self, test_samples, test_labels=None):
        if test_labels is None:
            return self.reg.predict(test_samples)
        else:
            return self.reg.score(test_samples, test_labels)
