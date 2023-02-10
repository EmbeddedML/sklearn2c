import os.path as osp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn2c.cppwriter import PolynomialRegExporter
from reg_data_generator import generate_regression_data

X, y, coeff1 = generate_regression_data(100, 2, 100, rs= 9)

linear_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
poly = PolynomialFeatures(3) # [1, x1,x2, x1^2, ]
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
fitted_reg = linear_reg.fit(X_train, y_train)
PolyRegWriter = PolynomialRegExporter(fitted_reg)
PolyRegWriter.export2file(osp.join('polynomial_regression', 'Inc', 'poly_reg_config.c'))