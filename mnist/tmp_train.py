import torch
import torchvision
import numpy as np
import utils
import time
import argparse
import pickle
import torch.optim as optim
from utils import get_logger
from dataset import MNIST_Dataset,get_dataloaders,data_transforms, split_name
from ournet import OurNet
import torch.nn.functional as F
import torch.nn as nn
from utils import multi_data_loader
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as transforms

num_epochs = 3
num_classes = 10
batch_size = 100
learning_rate = 0.001


# MNIST dataset
train_dataset_1 = MNIST_Dataset('MNIST_M',is_train = True,max_data_num =200000,transform=data_transforms[split_name[True]])
train_dataset_2 = MNIST_Dataset('MNIST',is_train = True,max_data_num =200000,transform=data_transforms[split_name[True]])
train_dataset_3 = MNIST_Dataset('SVHN',is_train = True,max_data_num =200000,transform=data_transforms[split_name[True]])


test_dataset = MNIST_Dataset('SYNTHDIGITS',is_train = False,max_data_num =200000,transform=data_transforms[split_name[False]])


# Data loader
train_loader_1 = torch.utils.data.DataLoader(dataset=train_dataset_1,
                                           batch_size=batch_size, 
                                           shuffle=True)

train_loader_2 = torch.utils.data.DataLoader(dataset=train_dataset_2,
                                           batch_size=batch_size, 
                                           shuffle=True)

train_loader_3 = torch.utils.data.DataLoader(dataset=train_dataset_3,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
# mnist_m = MNIST_Dataset('MNIST_M',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])

# train_loader = DataLoader(mnist_m[:10000], batch_size=batch_size, shuffle=True, num_workers=8)
# test_loader = DataLoader(mnist_m[10000:12000], batch_size=batch_size, shuffle=True, num_workers=8)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8*8*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print('outshape',out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader_1)
for epoch in range(num_epochs):
    model.train()
    for i,(images, labels) in enumerate(train_loader_1):
        images = images.to(device)
        #print(images.shape)
        labels = torch.tensor(labels,requires_grad=False, dtype=torch.long)
        labels = labels.to(device)
        #print(labels.shape)
        output = model(images)
        #print(output.shape)
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        
        loss =F.nll_loss(output, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

for epoch in range(num_epochs):
    model.train()
    for i,(images, labels) in enumerate(train_loader_2):
        images = images.to(device)
        #print(images.shape)
        labels = torch.tensor(labels,requires_grad=False, dtype=torch.long)
        labels = labels.to(device)
        #print(labels.shape)
        output = model(images)
        #print(output.shape)
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        
        loss =F.nll_loss(output, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

for epoch in range(num_epochs):
    model.train()
    for i,(images, labels) in enumerate(train_loader_3):
        images = images.to(device)
        #print(images.shape)
        labels = torch.tensor(labels,requires_grad=False, dtype=torch.long)
        labels = labels.to(device)
        #print(labels.shape)
        output = model(images)
        #print(output.shape)
        m = nn.LogSoftmax(dim=1)
        output = m(output)
        
        loss =F.nll_loss(output, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = torch.tensor(labels,requires_grad=False, dtype=torch.long)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 2000 test images: {} %'.format(100 * correct / total))




