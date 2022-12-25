from sklearn.neighbors import KNeighborsClassifier
from data_generator import generate_classes
from sklearn.model_selection import train_test_split
from cppwriter import KNNExporter

X, y = generate_classes(False)
knn_cls = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
knn_cls.fit(X_train, y_train)
knn_cls.predict(X_test)
knn_writer = KNNExporter(knn_cls)
knn_writer.export2file('knn_config.c')
