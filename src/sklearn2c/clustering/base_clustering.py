import joblib

class BaseClustering:
    def __init__(self, classifier) -> None:
        self.clus = classifier

    def train(self, train_samples, save_path, **kwargs):
        self.clus = self.clus.fit(train_samples, **kwargs)
        if save_path:
            joblib.dump(self, save_path)

    @staticmethod
    def load(filename:str):
        with open(filename, "rb") as joblib_file:
            model = joblib.load(joblib_file)
        return model

    def predict(self, test_samples, probs = True):
        if probs:
            return self.clus.predict_proba(test_samples)
        else:
            return self.clus.predict(test_samples)