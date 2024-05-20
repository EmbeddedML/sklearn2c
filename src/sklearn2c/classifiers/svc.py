from sklearn.svm import SVC

from .base_classifier import BaseClassifier
from .clf_exporter import SVMExporter

class SVMClassifier(BaseClassifier):
    def __init__(self, **kwargs):
       self.clf = SVC(**kwargs)
       super().__init__(self.clf)

    def train(self, train_samples, train_labels, save_path = None):
        super().train(train_samples, train_labels, save_path)

    def predict(self, test_samples):
        self.result = super().predict(test_samples, probs=False)
        return self.result
    
    def score(self, test_samples):
        return self.clf.decision_function(test_samples)
 
    

    @staticmethod
    def load(filename:str) -> "SVMClassifier":
        model = BaseClassifier.load(filename)
        if not isinstance(model, SVMClassifier):
            raise TypeError(f"Expected an object of type SVMClassifier, but got {type(model)} instead.")
        return model
        
    def export(self, filename = 'svc_config'):
        svm_writer = SVMExporter(self.clf)
        svm_writer.export(filename)
