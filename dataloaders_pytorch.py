import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as transforms
from dataset_pytorch import FoieDataset



def get_data_loaders(trainFolder,valFolder,csv_train,csv_val,train_batch_size,val_batch_size,kwargs):
    colmean=(0.0777,0.760,0.0659)
    colstddev=(0.1325,0.1393,0.1203)

    colmean_val=[0.091,0.918,0.0842]
    colstddev_val=[0.1566,0.1597,0.1467]

    compose_train=transforms.Compose([
                                transforms.Resize(500),
                                transforms.RandomAffine(degrees=20,translate=[0.2,0.2],shear=0.2),
                                transforms.RandomRotation(20),
                                transforms.RandomResizedCrop(224,scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                transforms.ToTensor(),
                                transforms.Normalize(colmean, colstddev)])

    compose_val=transforms.Compose([transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(colmean_val, colstddev_val)])
    

    train_loader = data.DataLoader(FoieDataset(imgFolder=trainFolder,csvPath=csv_train,transform=compose_train),
                    batch_size=train_batch_size,   shuffle=True, **kwargs)

    val_loader = data.DataLoader(FoieDataset(imgFolder=valFolder,csvPath=csv_val,transform=compose_val),
                    batch_size=val_batch_size,  shuffle=True, **kwargs)

    return train_loader, val_loader


