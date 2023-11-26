import joblib
from sklearn.tree import DecisionTreeClassifier

from .base_classifier import BaseClassifier
from .clf_writer import DTClassifierExporter

class DTClassifier(BaseClassifier):
    def __init__(self, **kwargs):
       self.clf = DecisionTreeClassifier(**kwargs)
       super().__init__(self.clf)

    def train(self, train_samples, train_labels, save = False):
        self.clf = super().train(train_samples, train_labels)
        if save:
            joblib.dump(self.clf, 'DecisionTree_classifier.joblib') 

    def inference(self, test_samples):
        result = super().inference(test_samples)
        return result
        
    def export(self, filename = 'dtc_config'):
        TreeWriter = DTClassifierExporter(self.clf)
        TreeWriter.export(filename)
