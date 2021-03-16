import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import dataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
import time

class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

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

def main():
    print("loading data")
    image_fnames, data_fnames = dataset.find_images()
    images, landmarks_2d, landmarks_3d = dataset.load_data(image_fnames, data_fnames, use_loading_bar=False)
    dataset.augment_flip(images, landmarks_2d, landmarks_3d)
    images = np.array(images)
    landmarks_2d = np.array(landmarks_2d)
    landmarks_3d = np.array(landmarks_3d)

    X_train, X_val, Y_train, Y_val = train_test_split(images, landmarks_2d, train_size=0.8, test_size=0.2)

    from torch.utils.data import DataLoader, TensorDataset

    # BATCH_SIZE = 16

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    # valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("data loaded")
    # train_data=TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(Y_train))
    # val_data=TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(Y_val))
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    # valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)
    train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader=DataLoader(valid_dataset, batch_size=64, shuffle=True)

    mfbs, label = next(iter(train_loader))
    # print("sample bacth:")

    print(mfbs.shape)
    print(label.shape)

    model = Net()
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model=model.cuda()
    # defining the optimizer
    # defining the loss function
        criterion=criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 100

    start_time = time.time()



    # checking if GPU is available
    # if torch.cuda.is_available():
    #     model = model.cuda(0)
    #     criterion = criterion.cuda(0)

    n_epochs = 10
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model
    for epoch in range(n_epochs):
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        model.train()
        for step in range(1,len(train_loader)+1):
            img, label = next(iter(train_loader))
            img = img.unsqueeze(1)  # if torch tensor
            label=label.view(label.size(0),-1)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                label=label.cuda()
                img=img.cuda()
            prediction=model(img.float())
            loss=criterion(prediction, label)
            loss.backward()
            optimizer.step()
            loss_train+=loss.item()
        t=time.time()
        runtime=t-prev_time
        train_loss.append(loss_train/len(train_loader))
    
        with torch.no_grad():
            for step in range(1,len(val_loader)+1):
                img, label = next(iter(val_loader))
                img=img.cuda()
                label=label.cuda()
                prediction=model(img)
                loss=criterion(prediction, label)
                loss_valid+=loss.item()
            valid_loss.append(loss_train/len(val_loader))

        if epoch%2==0:
            print("epoch=",epoch, "train_loss=",loss_train/len(train_loader),"valid_loss=", loss_valid/len(val_loader),"time=",runtime)
        prev_time=time.time()
    

    x_sample = x_val[0].numpy().copy()
    y_sample = y_val[0].numpy().copy()
    fig, ax = plt.subplots()
    y_out = model(x_sample).cpu().detach().numpy()[0]
    ax.imshow(x_sample, cmap="gray")
    ax.scatter(y_sample[:,0], y_sample[:,1])
    ax.scatter(y_out[0::2], y_out[1::2])
    plt.savefig("epoch%03d.png" % epoch)
    plt.close()


if __name__ == "__main__": 
    main()
