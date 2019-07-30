import torch
import torchvision
import matplotlib.pyplot as plt



def GenerativePlot(generator,Config,random=True):
    '''
    显示Generator生成的图片
    '''
    total_images = Config.total_images
    image_per_row = Config.image_per_row
    z_dim = generator.z_dim

    if random:
        z = 10*torch.randn(total_images,z_dim)
    else:
        z = torch.randn(1, z_dim)
        z = z.expand(total_images, -1).clone()
        Range = torch.linspace(-10, 10, image_per_row)
        for i in range(image_per_row):
            for j in range(image_per_row):
                z[i * image_per_row + j, 0] = Range[i]
                z[i * image_per_row + j, 1] = Range[j]

    if Config.use_gpu:
        z = z.to(Config.device)

    generative_images = generator(z)
    generative_images = generative_images.view(-1, 1, 28, 28)
    generative_images = torchvision.utils.make_grid(generative_images, nrow=image_per_row).permute(1, 2, 0).detach().cpu()
    plt.imshow(generative_images)
    plt.show()
    return generative_images
