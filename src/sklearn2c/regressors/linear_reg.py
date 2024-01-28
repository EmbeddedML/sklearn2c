from sklearn.linear_model import LinearRegression

from .base_regressor import BaseRegressor
from .reg_writer import PolynomialRegExporter

class LinearRegressor(BaseRegressor):
    def __init__(self, **kwargs):
       self.reg = LinearRegression(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save_path = None):
        self.reg = super().train(train_samples, train_labels, save_path)

    def predict(self, test_samples):
        result = super().predict(test_samples)
        return result
    
    @staticmethod
    def load(filename:str) -> "LinearRegressor":
        model = BaseRegressor.load(filename)
        if not isinstance(model, LinearRegressor):
            raise TypeError(f"Expected an object of type LinearRegressor, but got {type(model)} instead.")
        return model

    def export(self, filename = 'linReg_config'):
        LinearRegWriter = PolynomialRegExporter(self.reg)
        LinearRegWriter.export(filename)
