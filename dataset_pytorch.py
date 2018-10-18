import torch
from torch.utils.data import Dataset
import pandas as pd
import utils_pytorch as utils
from PIL import Image
import os
import numpy as np

class FoieDataset(Dataset):
    """
        Foie Dataset
    """
    def __init__(self,imgFolder,csvPath,transform=None):
        self.imgFolder=imgFolder
        self.files=os.listdir(imgFolder)
        self.csvPath=csvPath
        self.transform=transform
        self.df = pd.read_csv(csvPath, sep=';', encoding='utf-8')
        self.df['Typedelesion'].replace(to_replace='Kyste', value=0, inplace=True)
        self.df['Typedelesion'].replace(to_replace='Angiome', value=1, inplace=True)
        self.df['Typedelesion'].replace(to_replace='CHC', value=2, inplace=True)
        self.df['Typedelesion'].replace(to_replace='HNF', value=3, inplace=True)
        self.df['Typedelesion'].replace(to_replace='Metastase', value=4, inplace=True)
        self.df['Typedelesion'].replace(to_replace='Foie Homogene', value=5, inplace=True)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self,idx):
        row= self.df.loc[idx]
        filename1 = row.id
        img= Image.open(os.path.join(self.imgFolder, filename1 + '.bmp'), mode='r').convert("RGB")
        if self.transform:
            img = self.transform(img)
        class1 = row.Lesion
        if class1 == 1:
            class2=0
        else:
            class2=2
        class3 = row.Typedelesion
        return img,class1,class2,class3