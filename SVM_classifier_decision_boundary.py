import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from data_generator import X

model = SVC()
label = np.array([0] * 100 + [1] * 100)
model.fit(X, label)


xval, yval = np.meshgrid(np.arange(-10, 10, 0.05), np.arange(-10, 10, 0.05))

Z = model.decision_function(np.c_[xval.ravel(), yval.ravel()])

# Put the result into a color plot
# Create color maps
cmap_light = ListedColormap(['cornflowerblue','orange'])

Z = Z.reshape(xval.shape)

plt.figure(figsize=(7, 7))
plt.contourf(xval, yval, Z, cmap=cmap_light)

#fig, ax = plt.subplots()
#im = ax.imshow(Z, origin='lower', extent=[-10, 10, -10, 10])

plt.axis('Equal')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=label, edgecolors='k')

plt.show()
