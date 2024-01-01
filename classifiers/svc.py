import joblib
from sklearn.svm import SVC

from .base_classifier import BaseClassifier
from .clf_writer import SVMExporter

class SVMClassifier(BaseClassifier):
    def __init__(self, **kwargs):
       self.clf = SVC(**kwargs)
       super().__init__(self.clf)

    def train(self, train_samples, train_labels, save_path = None):
        self.clf = super().train(train_samples, train_labels, save_path)

    def inference(self, test_samples, test_labels = None):
        self.result = super().inference(test_samples, test_labels)
        return self.result
        
    def export(self, filename = 'svc_config'):
        svm_writer = SVMExporter(self.clf)
        svm_writer.export(filename)
