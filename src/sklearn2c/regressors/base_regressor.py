import joblib


class BaseRegressor:
    def __init__(self, regressor) -> None:
        self.reg = regressor

    def train(self, train_samples, train_labels, save_path, **kwargs):
        self.reg = self.reg.fit(train_samples, train_labels, **kwargs)
        if save_path:
            joblib.dump(self, save_path)
    
    @staticmethod
    def load(filename):
        with open(filename, "rb") as joblib_file:
            saved_model = joblib.load(joblib_file)
        return saved_model

    def predict(self, test_samples):
        return self.reg.predict(test_samples)
