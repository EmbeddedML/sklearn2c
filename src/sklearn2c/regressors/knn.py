from sklearn.neighbors import KNeighborsRegressor

from .base_regressor import BaseRegressor
from .reg_writer import KNNExporter

class KNNRegressor(BaseRegressor):
    def __init__(self, **kwargs):
       self.reg = KNeighborsRegressor(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save_path = None):
        self.reg = super().train(train_samples, train_labels, save_path)

    def predict(self, test_samples):
        result = super().predict(test_samples)
        return result

    @staticmethod
    def load(filename:str) -> "KNNRegressor":
        model = BaseRegressor.load(filename)
        if not isinstance(model, KNNRegressor):
            raise TypeError(f"Expected an object of type KNNRegressor, but got {type(model)} instead.")
        return model

    def export(self, filename = 'knnReg_config'):
        KNNRegWriter = KNNExporter(self.reg)
        KNNRegWriter.export(filename)