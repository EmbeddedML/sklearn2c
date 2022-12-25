from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

xval, yval = np.meshgrid(np.arange(-10, 10, 0.01), np.arange(-10, 10, 0.01))

# CLASS 1
MU_1 = [2.90542682, 2.04139237]
SIGMA_1_inv = np.array([[1.40938779, -0.04605597], [-0.04605597, 0.26668508]])
SIGMA_1_det = 0.61134405
CONST_1 = 1/(2*np.pi*np.sqrt(SIGMA_1_det))


Z1 = CONST_1*np.exp(-1/2*(SIGMA_1_inv[0, 0]*np.square(xval-MU_1[0]) + 2*SIGMA_1_inv[0, 1]
                         * (xval-MU_1[0])*(yval-MU_1[1]) + SIGMA_1_inv[1, 1]*np.square(yval-MU_1[1])))

# CLASS 2
MU_2 = [5.05639739, 1.03722113]
SIGMA_2_inv = np.array([[3.59164257, 0.23354834], [0.23354834, 1.31228639]])
SIGMA_2_det = 2.15840655
CONST_2 = 1/(2*np.pi*np.sqrt(SIGMA_2_det))

Z2 = CONST_2*np.exp(-1/2*(SIGMA_2_inv[0, 0]*np.square(xval-MU_2[0]) + 2*SIGMA_2_inv[0, 1]
                         * (xval-MU_2[0])*(yval-MU_2[1]) + SIGMA_2_inv[1, 1]*np.square(yval-MU_2[1])))

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xval, yval, Z2, cmap=cm.coolwarm)
ind2 = np.unravel_index(np.argmax(Z2, axis=None), Z2.shape)
print(xval[0,ind2[0]], yval[ind2[1],0])

ind1 = np.unravel_index(np.argmax(Z1, axis=None), Z1.shape)
print(xval[0,ind1[0]], yval[ind1[1],0])
Z3 = Z1 > Z2

fig, ax = plt.subplots()
im = ax.imshow(Z3, origin='lower', extent=[-10, 10, -10, 10])

plt.show()
