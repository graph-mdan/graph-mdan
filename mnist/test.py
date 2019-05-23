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
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(32),
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

cnn = ConvNet()
print(cnn.state_dict().keys())