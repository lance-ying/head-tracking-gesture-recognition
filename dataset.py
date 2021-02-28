import numpy as np
import scipy, scipy.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image
import os, os.path
from tqdm import tqdm

def find_images():
	# This will search the data directory to get the filenames of all images,
	# and their corresponding data filenames.
	#
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
	# We don't include the *_Flip directories, since it's easier to just flip the images in python

	image_fnames = []
	data_fnames = []
	for d in directories:
		raw_fnames = os.listdir(dataset_path + d)
		for fname in raw_fnames:
			if fname[-6:] == "_0.jpg":
				# We don't want to accidentally load an image multiple times, but there's a .jpg and .mat file
				# The _0 at the end ensures that it's not one of the "rotated" images included in the dataset
				image_fname = dataset_path + d + "/" + fname[:-4] + ".jpg"
				data_fname = dataset_path + "landmarks/" + d + "/" + fname[:-4] + "_pts.mat"
				image_fnames.append(image_fname)
				data_fnames.append(data_fname)

	print("Found %d images" % len(image_fnames))
	return image_fnames, data_fnames

def load_data(image_fnames, data_fnames):
	# Load the image and landmark data for each listed filename above.
	# Note that the indices must correspond -- don't shuffle them!
	print("Loading images...")
	images = [matplotlib.image.imread(image_fname) for image_fname in tqdm(image_fnames)]
	print("Loading landmarks...")
	datas = [scipy.io.loadmat(data_fname) for data_fname in tqdm(data_fnames)]
	landmarks_2d = [data["pts_2d"] for data in datas]
	landmarks_3d = [data["pts_3d"] for data in datas]
	print("Loaded %d images" % len(images))
	return images, landmarks_2d, landmarks_3d

if __name__ == "__main__":
	image_fnames, data_fnames = find_images()
	images, landmarks_2d, landmarks_3d = load_data(image_fnames, data_fnames)
	idx = np.random.randint(len(images))
	print("Displaying image: %s" % image_fnames[idx])
	print("            data: %s" % data_fnames[idx])

	img = images[idx]
	landmarks = landmarks_2d[idx]
	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.scatter(landmarks[:,0], landmarks[:,1])
	plt.show()