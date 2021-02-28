import numpy as np
import scipy, scipy.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

# filepath = "./data/300W_LP/AFW/AFW_1051618982_1_0"
# filepath = "./data/300W_LP/HELEN/HELEN_100032540_1_0"
# filepath = "./data/300W_LP/IBUG/IBUG_image_003_1_0"
filepath = "./data/300W_LP/LFPW/LFPW_image_test_0001_0"
img = matplotlib.image.imread(filepath + ".jpg")
data = scipy.io.loadmat(filepath + ".mat")
landmarks = data["pt2d"].T
landmarks[:,1] = img.shape[1] - landmarks[:,1] # LFPW only

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(landmarks[:,0], landmarks[:,1])
plt.show()