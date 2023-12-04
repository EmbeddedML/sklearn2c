import numpy as np
from sklearn.model_selection import train_test_split
from classification.svc import SVMClassifier

def Gaussian2D(mean,E,theta,len):
	c, s = np.cos(theta), np.sin(theta)
	R = np.array([[c, -s], [s, c]])
	cov=R@E@R.T
	return np.random.multivariate_normal(mean, cov, len).T

len=1000
mean = [-2, -2]
E = np.diag([1,10])
theta=np.radians(45)

x1, y1 =Gaussian2D(mean,E,theta,len)
cls1=np.vstack((x1, y1))


len=1000
mean = [2, 2]
E = np.diag([1,10])
theta=np.radians(-45)

x2, y2 = Gaussian2D(mean,E,theta,len)
cls2=np.vstack((x2, y2))

F=np.concatenate((cls1,cls2),axis=1)
F=F.T
labels = np.hstack((np.zeros(1000, dtype=np.uint8), np.ones(1000, dtype=np.uint8)))
train_samples, test_samples, train_labels, test_labels = train_test_split(F, labels, test_size=0.2)


svm = SVMClassifier()
svm.train(train_samples, train_labels)
svm.inference(test_samples)
# svm.export()
