from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from data_generator import generate_classes
from cppwriter import SVMExporter

X, y = generate_classes(False)
clf = SVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
fitted_svm = clf.fit(X_train, y_train)
SVMWriter = SVMExporter(fitted_svm)
SVMWriter.export2file('svm_config.c')
