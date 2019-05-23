import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from mnist import Mnist
from mnist_m import MnistM
from svhn import Svhn
from synthdigits import SynthDigits
import numpy as np
from PIL import Image


class MNIST_Dataset(Dataset): 

    def __init__(self, mnist_type,is_train,max_data_num,transform):
        self.is_train = is_train
        if mnist_type == "MNIST":
            self.data_type = Mnist(is_train, max_data_num)        
        elif mnist_type == "MNIST_M":            
            self.data_type = MnistM(is_train, max_data_num)        
        elif mnist_type == "SVHN":            
            self.data_type = Svhn(is_train, max_data_num)   
        elif mnist_type == "SYNTHDIGITS":   
            self.data_type = SynthDigits(is_train, max_data_num)
        #self.data_type = np.swapaxes(self.data_type, 0,2)
        self.file_size = self.data_type.file_size
        self.transform = transform

    def __len__(self): 
        return self.file_size

    def __getitem__(self, index): 
        image, label = self.data_type.image[index], self.data_type.label[index]
        #change from NHWC to NCHW
        
        image = np.transpose(image,(2,0,1))
        
        image = torch.from_numpy(image)
        
        image = self.transform(image)
        
        
        
        return image,label



data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(30),           
            transforms.ColorJitter(brightness=0.2, contrast=[0.5,1.5], saturation=0, hue=0),
            transforms.ToTensor(),]),
        'test': transforms.Compose([  
            transforms.ToPILImage(),                      
            transforms.CenterCrop(30),           
            transforms.ToTensor(),                  
        ]),   
    } 
     
split_name = {True: 'train', False:'test'}

def get_dataloaders(mnist_type,is_train,max_data_num, batch_size, shuffle, num_workers=32):
    
    image_dataset = MNIST_Dataset(mnist_type,is_train,max_data_num,transform=data_transforms[split_name[is_train]])    
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)



if __name__=="__main__":
    
    myset_loader = get_dataloaders(mnist_type='SVHN',is_train=True,max_data_num=20000, batch_size=10, shuffle=True)
    for i, (image,label)in enumerate(myset_loader):
        if i>5:
            break
        else:

            print(image.shape)
            print(label.shape)