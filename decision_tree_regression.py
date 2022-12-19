from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from cppwriter import DTRegressorExporter
from reg_data_generator import generate_regression_data

X, y, coeff1 = generate_regression_data(100, 2, 100, rs= 9)
reg = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
fitted_reg = reg.fit(X_train, y_train)
DTRegressorWriter = DTRegressorExporter(fitted_reg)
DTRegressorWriter.export2file('reg_tree_config.h')
result = fitted_reg.predict([[0.3, 0.8]])
print(result)