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

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.CONV1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
		self.CONV2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
		self.CONV3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
		self.CONV4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
		self.CONV5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
		self.CONV6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
		self.FC1=nn.Linear(6400,1024)
		self.FC2=nn.Linear(1024,136)
		self.POOL = nn.MaxPool2d(2,2)
		self.DROP = nn.Dropout(p=0.2)

	def forward(self, h):
		# print(h.detach().numpy().shape)
		h = self.CONV1(h)
		h = F.relu(h)
		h = self.POOL(h)
		h = self.DROP(h)

		h = self.CONV2(h)
		h = F.relu(h)
		h = self.POOL(h)
		h = self.DROP(h)

		h = self.CONV3(h)
		h = F.relu(h)
		h = self.POOL(h)
		h = self.DROP(h)

		h = self.CONV4(h)
		h = F.relu(h)
		h = self.POOL(h)
		h = self.DROP(h)

		h = self.CONV5(h)
		h = F.relu(h)
		h = self.POOL(h)
		h = self.DROP(h)

		h = self.CONV6(h)
		h = F.relu(h)
		h = self.POOL(h)
		h = self.DROP(h)
		# print(h.detach().numpy().shape)

		h = h.view(h.size(0), -1)

		h = self.FC1(h)
		h = F.relu(h)
		h = self.DROP(h)
		h = self.FC2(h)

		return h

def load_from_checkpoint(fname):
	dic = torch.load(fname)
	model = Net()
	model.load_state_dict(dic)
	return model

def torchify_image(image):
	try:
		image.reshape(1,1,450,450)
	except ValueError:
		print("Image must be 450x450 resolution. Actual image shape was %s" % str(image.shape))
		exit(1)
	return torch.from_numpy(image.reshape(1,1,450,450)).type(torch.float)

def detorchify_output(output):
	return output.cpu().detach().numpy()[0]