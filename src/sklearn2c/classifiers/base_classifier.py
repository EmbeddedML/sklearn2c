import joblib

class BaseClassifier:
    def __init__(self, classifier) -> None:
        self.clf = classifier

    def train(self, train_samples, train_labels, save_path, **kwargs):
        self.clf = self.clf.fit(train_samples, train_labels, **kwargs)
        if save_path:
            joblib.dump(self, save_path)

    @staticmethod
    def load(filename:str):
        with open(filename, "rb") as joblib_file:
            model = joblib.load(joblib_file)
        return model

    def predict(self, test_samples, probs = False):
        if probs:
            return self.clf.predict_proba(test_samples)
        else:
            return self.clf.predict(test_samples)
