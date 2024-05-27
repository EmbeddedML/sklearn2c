from sklearn.tree import DecisionTreeRegressor
from .base_regressor import BaseRegressor
from .reg_exporter import DTRegressorExporter

class DTRegressor(BaseRegressor):
    def __init__(self, **kwargs):
       self.reg = DecisionTreeRegressor(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save_path = None):
        super().train(train_samples, train_labels, save_path)

    def predict(self, test_samples):
        result = super().predict(test_samples)
        return result
    
    @staticmethod
    def load(filename:str) -> "DTRegressor":
        model = BaseRegressor.load(filename)
        if not isinstance(model, DTRegressor):
            raise TypeError(f"Expected an object of type DTRegressor, but got {type(model)} instead.")
        return model
        
    def export(self, filename = 'dtr_config'):
        TreeWriter = DTRegressorExporter(self.reg)
        TreeWriter.export(filename)
