from sklearn.linear_model import LinearRegression

from .base_regressor import BaseRegressor
from .reg_writer import PolynomialRegExporter

class LinearRegressor(BaseRegressor):
    def __init__(self, **kwargs):
       self.reg = LinearRegression(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save_path = None):
        self.reg = super().train(train_samples, train_labels, save_path)

    def inference(self, test_samples, test_labels = None):
        result = super().inference(test_samples, test_labels)
        return result

    def export(self, filename = 'linReg_config'):
        LinearRegWriter = PolynomialRegExporter(self.reg)
        LinearRegWriter.export(filename)
