import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule



class Generator_fc(BasicModule):

    def __init__(self,image_size=28*28,hidden_dim=400,z_dim=20):
        super(Generator_fc,self).__init__()

        self.model_name = 'generator_fc'
        self.z_dim = z_dim

        self.fc1 = nn.Linear(z_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,image_size)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = torch.sigmoid(self.fc2(h))
        return out


class Generator_conv(BasicModule):

    def __init__(self,image_size=28,channel=1,z_dim=64):
        super(Generator_conv,self).__init__()

        self.model_name = 'generator_conv'
        self.z_dim = z_dim

        self.fc = nn.Sequential(
            nn.Linear(z_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,128*7*7),
            nn.BatchNorm1d(128*7*7),
            nn.ReLU()
        )
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h1 = self.fc(x)
        h2 = self.dconv(h1.view(-1,128,7,7))
        return h2



class Discriminator_fc(BasicModule):

    def __init__(self,image_size=28*28,hidden_dim=400):
        super(Discriminator_fc,self).__init__()

        self.model_name = 'discriminator_fc'

        self.fc1 = nn.Linear(image_size,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        p = torch.sigmoid(self.fc2(h))
        return p


class Discriminator_conv(BasicModule):

    def __init__(self,image_size=28,channel=1):
        super(Discriminator_conv,self).__init__()

        self.model_name = 'discriminator_conv'

        self.conv = nn.Sequential(
            nn.Conv2d(1,64,4,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h1 = self.conv(x)
        h2 = self.fc(h1.view(-1,128*7*7))
        return h2