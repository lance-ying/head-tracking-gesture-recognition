import numpy as np
import scipy, scipy.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

filepath = "./data/300W_LP/AFW/AFW_1051618982_1_0"
img = matplotlib.image.imread(filepath + ".jpg")
data = scipy.io.loadmat(filepath + ".mat")
landmarks = data["pt2d"].T

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(landmarks[:,0], landmarks[:,1])
plt.show()