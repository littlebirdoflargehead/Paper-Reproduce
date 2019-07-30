import torch



def D_loss(real_images,generative_images,discriminator):
    loss = torch.log(discriminator(real_images))+torch.log(1-discriminator(generative_images))
    loss = -torch.mean(loss)
    return loss

def G_loss(generative_images,discriminator):
    loss = torch.log(discriminator(generative_images))
    loss = -torch.mean(loss)
    return loss