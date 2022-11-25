from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from data_generator import generate_classes
from cppwriter import DTRegressorExporter

X, y = generate_classes(False)
reg = DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
fitted_reg = reg.fit(X_train, y_train)
DTRegressorWriter = DTRegressorExporter(fitted_reg)
DTRegressorWriter.export2file('dt_reg_config.c')
