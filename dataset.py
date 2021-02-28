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
directories = ["AFW", "HELEN", "IBUG", "LFPW"]

image_fnames = []
data_fnames = []
for d in directories:
	raw_fnames = os.listdir(dataset_path + d)
	for fname in raw_fnames:
		if fname[-6:] == "_0.jpg":
			image_fname = dataset_path + d + "/" + fname[:-4] + ".jpg"
			data_fname = dataset_path + "/landmarks/" + d + "/" + fname[:-4] + "_pts.mat"
			image_fnames.append(image_fname)
			data_fnames.append(data_fname)

print("Found %d images" % len(image_fnames))

# idx = 0
idx = np.random.randint(len(image_fnames))
img = matplotlib.image.imread(image_fnames[idx])
data = scipy.io.loadmat(data_fnames[idx])
landmarks = data["pts_2d"]
print("Image filename: %s" % image_fnames[idx])
print("Data filename:  %s" % data_fnames[idx])

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(landmarks[:,0], landmarks[:,1])
plt.show()