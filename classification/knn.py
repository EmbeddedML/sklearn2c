import joblib
from sklearn.neighbors import KNeighborsClassifier
from .base_classifier import BaseClassifier
from .clf_writer import KNNExporter

class KNNClassifier(BaseClassifier):
    def __init__(self, **kwargs):
       self.clf = KNeighborsClassifier(**kwargs)
       super().__init__(self.clf)

    def train(self, train_samples, train_labels, save = False):
        self.clf = super().train(train_samples, train_labels)
        if save:
            joblib.dump(self.clf, 'DecisionTree_classifier.joblib') 

    def inference(self, test_samples):
        self.result = super().inference(test_samples)
        return self.result
        
    def export(self, filename = 'knn_config'):
        TreeWriter = KNNExporter(self.clf)
        TreeWriter.export(filename)
