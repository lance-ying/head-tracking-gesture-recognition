import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from torch.autograd import Variable
import dataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import dataset
from network import Net

def random_translate(image, landmarks, translation_pixel_padding = 5, roll_overwrite_zero = True):
	minx = int(np.floor(np.min(landmarks[:,0])))
	miny = int(np.ceil(np.min(landmarks[:,1])))
	maxx = int(np.floor(np.max(landmarks[:,0])))
	maxy = int(np.ceil(np.max(landmarks[:,1])))
	lx = -minx + translation_pixel_padding
	ly = -miny + translation_pixel_padding
	hx = image.shape[1] - maxx - translation_pixel_padding
	hy = image.shape[0] - maxy - translation_pixel_padding
	dx = np.random.randint(lx, hx) if lx < hx else 0
	dy = np.random.randint(ly, hy) if ly < hy else 0
	image = np.roll(image, (dy,dx), axis=(0,1))
	if roll_overwrite_zero:
		if dx > 0:
			image[:,0:dx] = 0
		if dx < 0:
			image[:,dx:] = 0
		if dy > 0:
			image[0:dy,:] = 0
		if dy < 0:
			image[dy:,:] = 0
	landmarks[:,0] += dx
	landmarks[:,1] += dy
	return image, landmarks

def main(batch_size = 64, use_gpu = False, train_size = 0.8, test_size = 0.2, use_loading_bar = True, learning_rate = 0.0001, num_epochs = 5, epoch_print = 1, epoch_save = 5, translation_pixel_padding = 5, roll_overwrite_zero = True, checkpoint_dir = "checkpoints/"):
	image_fnames, data_fnames = dataset.find_images()
	images, landmarks_2d, landmarks_3d = dataset.load_data(image_fnames, data_fnames, use_loading_bar=use_loading_bar)
	dataset.augment_flip(images, landmarks_2d, landmarks_3d)
	images = np.array(images)
	landmarks_2d = np.array(landmarks_2d)
	landmarks_3d = np.array(landmarks_3d)

	X_train, X_val, Y_train, Y_val = train_test_split(images, landmarks_2d, train_size=train_size, test_size=test_size)
	train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
	valid_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
	train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

	model = Net()
	criterion = nn.MSELoss()

	if use_gpu and torch.cuda.is_available():
		model = model.cuda()
		criterion = criterion.cuda()
		using_gpu = True
	else:
		using_gpu = False

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	loss_min = np.inf
	train_loss = []
	valid_loss = []

	start_time = time.time()

	for epoch in range(num_epochs):
		prev_time = time.time()

		loss_train = 0
		loss_valid = 0
		running_loss = 0
		model.train()
		for step in (tqdm(range(1,len(train_loader)+1)) if use_loading_bar else range(1,len(train_loader)+1)):
			img, label = next(iter(train_loader))
			img = img.numpy().astype(np.float32)/255
			landmarks = label.numpy()
			m = np.mean(img, axis=(1,2))
			s = np.std(img, axis=(1,2))
			img = (img - m[:,None,None]) / s[:,None,None]
			for i in range(len(img)):
				img[i], landmarks[i] = random_translate(img[i], landmarks[i], translation_pixel_padding=translation_pixel_padding, roll_overwrite_zero=roll_overwrite_zero)
			img = torch.tensor(img).unsqueeze(1)
			label = torch.tensor(landmarks)
			label = label.view(label.size(0),-1)

			optimizer.zero_grad()
			if using_gpu:
				label = label.cuda()
				img = img.cuda()

			prediction = model(img)
			loss = criterion(prediction, label)
			
			loss.backward()
			optimizer.step()
			loss_train += loss.item()

		t = time.time()
		runtime = t - prev_time
		train_loss.append(loss_train / len(train_loader))

		with torch.no_grad():
			for step in range(1, len(val_loader)+1):
				img, label = next(iter(val_loader))
				img = img.numpy().astype(np.float32)/255
				m = np.mean(img, axis=(1,2))
				s = np.std(img, axis=(1,2))
				img = (img - m[:,None,None]) / s[:,None,None]
				for i in range(len(img)):
					img[i], landmarks[i] = random_translate(img[i], landmarks[i], translation_pixel_padding=translation_pixel_padding, roll_overwrite_zero=roll_overwrite_zero)
				img = img.unsqueeze(1)
				label = label.view(label.size(0),-1)

				if using_gpu:
					img = img.cuda()
					label = label.cuda()

				prediction = model(img)
				loss = criterion(prediction, label)
				loss_valid += loss.item()
			valid_loss.append(loss_train / len(val_loader))

		if epoch % epoch_print == 0:
			print("epoch=", epoch, "train_loss=", loss_train/len(train_loader), "valid_loss=", loss_valid/len(val_loader), "time=", runtime)

		if epoch % epoch_save == 0 or epoch + 1 == num_epochs:
			state = {
				"epoch": epoch,
				"state_dict": model.state_dict(),
			}
			filename = os.path.join(os.getcwd(), checkpoint_dir, (str(epoch) + ".checkpoint"))
			torch.save(model.state_dict(), filename)

if __name__ == "__main__":
	main()