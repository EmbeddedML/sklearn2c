import os.path as osp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from data_generator import MLClass, generate_classes
from cppwriter import PolynomialRegExporter

LABEL_TO_IDX = {"CLASS 1":0, "CLASS 2":1}
MEAN_1 = [2.5, 2.]
STD_DEV1 = [1, 2]
MEAN_2 = [1, 2]
STD_DEV2 = [.5, 1]
ml_class1 = MLClass("CLASS 1", 100, MEAN_1, STD_DEV1)
ml_class2 = MLClass("CLASS 2", 100, MEAN_2, STD_DEV2)
all_classes = [ml_class1, ml_class2]
samples, labels = generate_classes(all_classes)

reg = LinearRegression()
labels = [LABEL_TO_IDX[label] for label in labels]
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.5, random_state=0)
fitted_reg = reg.fit(X_train, y_train)
PolyRegWriter = PolynomialRegExporter(fitted_reg)
PolyRegWriter.export2file(osp.join('polynomial_regression', 'Inc', 'poly_reg_config.c'))