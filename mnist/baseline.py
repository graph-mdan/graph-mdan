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




# MNIST dataset
train_dataset_1 = MNIST_Dataset('MNIST_M',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])
train_dataset_2 = MNIST_Dataset('MNIST',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])
train_dataset_3 = MNIST_Dataset('SVHN',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])
train_dataset_4 = MNIST_Dataset('SYNTHDIGITS',is_train = True,max_data_num =20000,transform=data_transforms[split_name[True]])


#test_dataset = MNIST_Dataset('SYNTHDIGITS',is_train = False,max_data_num =200000,transform=data_transforms[split_name[False]])
test_dataset = MNIST_Dataset('MNIST',is_train = False,max_data_num =20000,transform=data_transforms[split_name[False]])



## baseline model

class Baseline(nn.Module):
    def __init__(self,configs):
        super(Baseline, self).__init__()
        self.device = configs['device']
        self.num_classes = configs['num_classes']
        self.batch_size = configs["batch_size"]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.target_classifier = nn.Sequential(
            
            nn.Linear(3200, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1,3200)
        x = self.target_classifier(x)

        return x

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
        self.fc = nn.Linear(1568, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print('outshape',out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

#model = ConvNet(10).to(device)

### Hyperparameters   
configs = {"num_classes": 10,
               "num_epochs": 15, "batch_size": 32, "lr": 0.001,  'device':device, 
                }   

model = Baseline(configs).to(device)
#Data loader
train_loader_1 = torch.utils.data.DataLoader(dataset=train_dataset_1,
                                           batch_size=configs['batch_size'], 
                                           shuffle=True)

train_loader_2 = torch.utils.data.DataLoader(dataset=train_dataset_2,
                                           batch_size=configs['batch_size'], 
                                           shuffle=True)

train_loader_3 = torch.utils.data.DataLoader(dataset=train_dataset_3,
                                           batch_size=configs['batch_size'],
                                           shuffle=True)

train_loader_4 = torch.utils.data.DataLoader(dataset=train_dataset_4,
                                           batch_size=configs['batch_size'],
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=configs['batch_size'],
                                          shuffle=False)



learning_rate = configs['lr']
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = configs['num_epochs']

# training
total_step = len(train_loader_4) ## don't make mistake here
for epoch in range(configs['num_epochs']):
    model.train()
    for i,(images, labels) in enumerate(train_loader_4):
        images = images.to(device)
        #print(images.shape)
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
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    count = 0
    for images, labels in test_loader:
        # count +=1
        # if count <10:
        #     print(labels)
        images = images.to(device)
        labels = torch.tensor(labels,requires_grad=False, dtype=torch.long)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 20000 test images: {} %'.format(100 * correct / total))
