import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .base_regressor import BaseRegressor
from .reg_writer import PolynomialRegExporter

class PolynomialRegressor(BaseRegressor):
    def __init__(self, deg, **kwargs):
       self.poly_features = PolynomialFeatures(degree = deg, **kwargs)
       self.reg = LinearRegression(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save = False):
        train_samples = self.poly_features.fit_transform(train_samples)
        self.reg = super().train(train_samples, train_labels)
        if save:
            joblib.dump(self, 'poly_regression.joblib') 

    def inference(self, test_samples, test_labels = None):
        test_samples = self.poly_features.fit_transform(test_samples)
        result = super().inference(test_samples, test_labels)
        return result

    def export(self, filename = 'polyReg_config'):
        feature_names = self.poly_features.get_feature_names_out()
        PolyWriter = PolynomialRegExporter(self.reg, feature_names)
        PolyWriter.export(filename)
