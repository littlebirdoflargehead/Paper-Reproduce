import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule



class VampPriorVAE(BasicModule):

    def __init__(self,image_size=28*28,hidden_dim=400,z_dim=20,K=64):
        super(VampPriorVAE,self).__init__()

        self.model_name = 'vpvae'

        self.z_dim = z_dim

        self.fc1 = nn.Linear(image_size,hidden_dim)
        self.fc21 = nn.Linear(hidden_dim,z_dim)
        self.fc22 = nn.Linear(hidden_dim,z_dim)
        self.fc3 = nn.Linear(z_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,image_size)

        self.idle_input = nn.parameter.Parameter(torch.eye(K),requires_grad=False)
        self.pseudo = nn.Linear(K,image_size)

    def encoder(self,x):
        h1 = self.fc1(x)
        mu = self.fc21(F.relu(h1))
        logvar = self.fc22(F.relu(h1))
        return mu,logvar

    def decoder(self,z):
        h2 = self.fc3(z)
        x = self.fc4(F.relu(h2))
        return torch.sigmoid(x)

    def reparameter_trick(self,mu,logvar):
        epslon = torch.randn_like(logvar)
        std = torch.exp(0.5*logvar)
        z = mu+epslon*std
        return z,epslon

    def pseudo_image(self):
        pseudo_images = torch.sigmoid(self.pseudo(self.idle_input))
        return pseudo_images

    def forward(self, x):
        mu,logvar = self.encoder(x)
        z,_ = self.reparameter_trick(mu,logvar)
        out = self.decoder(z)
        return out,mu,logvar