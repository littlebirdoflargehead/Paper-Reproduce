import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule



class Generative_fc(BasicModule):

    def __init__(self,image_size=28*28,hidden_dim=400,z_dim=20):
        super(Generative_fc,self).__init__()

        self.model_name = 'generative'

        self.fc1 = nn.Linear(z_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,image_size)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = torch.sigmoid(self.fc2(h))
        return out



class Discriminator_fc(BasicModule):

    def __init__(self,image_size=28*28,hidden_dim=400):
        super(Discriminator_fc,self).__init__()

        self.model_name = 'discriminator'

        self.fc1 = nn.Linear(image_size,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        p = torch.sigmoid(self.fc2(h))
        return p