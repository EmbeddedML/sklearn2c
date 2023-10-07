from classification.data_generator import generate_classes, MLClass
from classification.bayes import BayesClassifier
from classification.knn import KNNClassifier
from classification.dtc import DTClassifier
from classification.svc import SVMClassifier
from sklearn.model_selection import train_test_split

MEAN_1 = [2.5, 2.]
STD_DEV1 = [1, 2]
MEAN_2 = [1, 2]
STD_DEV2 = [.5, 1]
ml_class1 = MLClass("CLASS 1", 100, MEAN_1, STD_DEV1)
ml_class2 = MLClass("CLASS 2", 100, MEAN_2, STD_DEV2)
all_classes = [ml_class1, ml_class2]
samples, labels = generate_classes(all_classes)
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.2, random_state=42)

bayesian = BayesClassifier()
bayesian.train(train_samples, train_labels)
bayesian.inference(test_samples)
bayesian.export()

dtc = DTClassifier()
dtc.train(train_samples, train_labels)
dtc.inference(test_samples)
dtc.export()

knn = KNNClassifier()
knn.train(train_samples, train_labels)
knn.inference(test_samples)
knn.export()

svm = SVMClassifier()
svm.train(train_samples, train_labels)
svm.inference(test_samples)
svm.export()