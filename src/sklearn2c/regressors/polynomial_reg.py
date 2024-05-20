from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .base_regressor import BaseRegressor
from .reg_exporter import PolynomialRegExporter

class PolynomialRegressor(BaseRegressor):
    def __init__(self, deg = 2, **kwargs):
       self.poly_features = PolynomialFeatures(degree = deg, **kwargs)
       self.reg = LinearRegression(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save_path = None):
        self.num_inputs = train_samples.shape[1]
        train_samples = self.poly_features.fit_transform(train_samples)
        self.reg = super().train(train_samples, train_labels, save_path)

    def predict(self, test_samples):
        test_samples = self.poly_features.fit_transform(test_samples)
        result = super().predict(test_samples)
        return result
    
    @staticmethod
    def load(filename:str) -> "PolynomialRegressor":
        model = BaseRegressor.load(filename)
        if not isinstance(model, PolynomialRegressor):
            raise TypeError(f"Expected an object of type PolynomialRegressor, but got {type(model)} instead.")
        return model

    def export(self, filename = 'polyReg_config'):
        feature_names = self.poly_features.get_feature_names_out()
        PolyWriter = PolynomialRegExporter(self.reg, self.num_inputs, feature_names)
        PolyWriter.export(filename)
