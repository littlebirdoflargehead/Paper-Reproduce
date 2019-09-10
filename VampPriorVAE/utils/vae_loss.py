import torch
import torch.nn.functional as F



def VAE_Loss(images,re_images,mu,logvar):
    BCE = F.binary_cross_entropy(re_images,images,reduction='sum')
    s = 1+logvar-torch.pow(mu,2)-torch.exp(logvar)
    KLD = 0.5*torch.sum(s)
    return BCE - KLD



def VampPriorVAE_Loss(images,model):
    mu,logvar = model.encoder(images)
    z, _ = model.reparameter_trick(mu, logvar)
    re_images = model.decoder(z)

    pseudo_images = model.pseudo_image()
    mu_k, logvar_k = model.encoder(pseudo_images)
    PCE = 0
    for j in range(len(z)):
        z_j = z[j].expand(mu_k.shape[0],-1)
        logq_k = torch.pow(z_j-mu_k,2)/torch.exp(logvar_k)+logvar_k
        logq_k = torch.sum(torch.exp(-0.5*logq_k),dim=1)
        PCE += torch.log(torch.mean(logq_k))

    BCE = F.binary_cross_entropy(re_images,images,reduction='sum')
    QE = 0.5*torch.sum(1+logvar)
    return BCE - PCE - QE