class BaseClassifier:
    def __init__(self, classifier) -> None:
        self.clf = classifier
    def train(self, train_samples, train_labels, **kwargs):
        return self.clf.fit(train_samples, train_labels, **kwargs)
    def inference(self, test_samples, test_labels = None):
        if test_labels is None:
            return self.clf.predict(test_samples)
        else:
            return self.clf.score(test_samples, test_labels)
        