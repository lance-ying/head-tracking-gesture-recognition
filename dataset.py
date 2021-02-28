import numpy as np
import scipy, scipy.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image
import os, os.path

# Directory structure should be
# data
# |-- 300W_LP
# |   |-- AFW
# |   |   |-- AFW_#####.jpg
# |   |   |-- AFW_#####.mat
# |   |
# |   |-- AFW_Flip
# |   |   |-- AFW_#####.jpg
# |   |   |-- AFW_#####.mat

dataset_path = "./data/300W_LP/"

directories_partial = ["AFW", "HELEN", "IBUG", "LFPW"]
directories = []
for d in directories_partial:
	directories.append(d)
	directories.append(d + "_Flip")

image_names = []
for d in directories:
	raw_fnames = os.listdir(dataset_path + d)
	for fname in raw_fnames:
		if fname[-6:] == "_0.jpg":
			image_names.append(dataset_path + d + "/" + fname[:-4])

print("Found %d images" % len(image_names))

# filepath = image_names[0]
filepath = image_names[np.random.randint(len(image_names))]
# filepath = "./data/300W_LP/AFW/AFW_1051618982_1_0"
# filepath = "./data/300W_LP/HELEN/HELEN_100032540_1_0"
# filepath = "./data/300W_LP/IBUG/IBUG_image_003_1_0"
# filepath = "./data/300W_LP/LFPW/LFPW_image_test_0001_0"
img = matplotlib.image.imread(filepath + ".jpg")
data = scipy.io.loadmat(filepath + ".mat")
landmarks = data["pt2d"].T
# landmarks[:,1] = img.shape[1] - landmarks[:,1] # LFPW only

# temp = "AFW_1051618982_1_0"
# img = matplotlib.image.imread("./data/300W_LP/AFW/" + temp + ".jpg")
# data = scipy.io.loadmat("./data/300W_LP/landmarks/AFW/" + temp + "_pts.mat")
# landmarks = data["pts_2d"]

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(landmarks[:,0], landmarks[:,1])
plt.show()