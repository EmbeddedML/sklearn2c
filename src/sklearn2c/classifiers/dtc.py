import joblib
from sklearn.tree import DecisionTreeClassifier

from .base_classifier import BaseClassifier
from .clf_exporter import DTClassifierExporter

class DTClassifier(BaseClassifier):
    def __init__(self, **kwargs):
       self.clf = DecisionTreeClassifier(**kwargs)
       super().__init__(self.clf)

    def train(self, train_samples, train_labels, save_path = None):
        super().train(train_samples, train_labels, save_path)

    def predict(self, test_samples):
        result = super().predict(test_samples)
        return result
    
    @staticmethod
    def load(filename:str) -> "DTClassifier":
        model = BaseClassifier.load(filename)
        if not isinstance(model, DTClassifier):
            raise TypeError(f"Expected an object of type DTClassifier, but got {type(model)} instead.")
        return model
        
    def export(self, filename = 'dtc_config'):
        TreeWriter = DTClassifierExporter(self.clf)
        TreeWriter.export(filename)
