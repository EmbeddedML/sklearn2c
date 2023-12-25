import joblib


class BaseClassifier:
    def __init__(self, classifier) -> None:
        self.clf = classifier

    def train(self, train_samples, train_labels, save_path, **kwargs):
        self.clf = self.clf.fit(train_samples, train_labels, **kwargs)
        if save_path:
            joblib.dump(self.clf, save_path)
        return self.clf

    def load(self, filename):
        with open(filename, "rb") as pkl_file:
            self.clf = joblib.load(pkl_file)

    def inference(self, test_samples, probs=True):
        if probs:
            return self.clf.predict_proba(test_samples)
        else:
            return self.clf.predict(test_samples)
