import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
class Model1(nn.Module):
    def __init__(self,classes1,classes2,classes3):
        super(Model1, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.last_layer= torch.nn.Sequential(*list(self.base_model.children())[:-2])
        ct = 0
        for child in self.last_layer.children():
            ct += 1
            if ct < 25:
                for param in child.parameters():
                    param.requires_grad = False
        #for p in self.base_model.parameters():
        #    p.requires_grad = False

        self.groupfinal = nn.Sequential(
            nn.Linear(2048*7*7,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.groupSain = nn.Sequential(
            
            nn.Linear(128,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16,classes1),
            nn.Softmax()
        )
        self.groupMalin = nn.Sequential(
             nn.Linear(128,16),
             nn.BatchNorm1d(16),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(16,classes2),
             nn.Softmax()
        )
        self.groupAnomaly = nn.Sequential(
             nn.Linear(128,32),
             nn.BatchNorm1d(32),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(32,classes3),
             nn.Softmax()
        )

        
    def forward(self, x):
        #out=self.base_model(x)
        out2= self.last_layer(x)
        out3= out2.view(out2.size(0), -1)
        out4= self.groupfinal(out3)
        outa,outb,outc=self.groupSain(out4),self.groupMalin(out4),self.groupAnomaly(out4)
        return outa,outb,outc



