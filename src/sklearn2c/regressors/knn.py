import joblib
from sklearn.neighbors import KNeighborsRegressor

from .base_regressor import BaseRegressor
from .reg_writer import KNNExporter

class KNNRegressor(BaseRegressor):
    def __init__(self, **kwargs):
       self.reg = KNeighborsRegressor(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save_path = None):
        self.reg = super().train(train_samples, train_labels, save_path)

    def inference(self, test_samples, test_labels = None):
        result = super().inference(test_samples, test_labels)
        return result

    def export(self, filename = 'knnReg_config'):
        KNNRegWriter = KNNExporter(self.reg)
        KNNRegWriter.export(filename)