__version__ = "0.0.1"
__doc__ = "A simple tool to embed scikit-learn models into microcontrollers"

from .classifiers import BayesClassifier, DTClassifier, KNNClassifier, SVMClassifier
from .regressors import LinearRegressor, PolynomialRegressor, DTRegressor, KNNRegressor

def main():
    print("Main function called")