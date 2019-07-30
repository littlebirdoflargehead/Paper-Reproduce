import torch
import models
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from utils import GenerativePlot,G_loss,D_loss
from config import Config



def train(Config):
    '''
    模型训练的整个流程，包括：
    step1: 数据
    step2: 定义模型
    step3: 目标函数与优化器
    step4: 统计指标：平滑处理之后的损失，还有混淆矩阵（无监督训练时不需要）
    训练并统计
    '''

    # step1: 数据
    train_dataset = torchvision.datasets.MNIST(root=Config.train_data_root,train=True,transform=transforms.ToTensor(),download=False)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=Config.batch_size,shuffle=True,num_workers=Config.num_workers)

    # step2: 定义模型
    generator = models.Generative_fc(z_dim=2)
    discriminator = models.Discriminator_fc()
    if Config.load_model_path:
        generator.load(Config.load_model_path)
        discriminator.load(Config.load_model_path)
    if Config.use_gpu:
        generator.to(Config.device)
        discriminator.to(Config.device)

    # step3: 目标函数与优化器

    lr = Config.lr
    optimizer_g = torch.optim.Adam(generator.parameters(),lr=10*lr,weight_decay=Config.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(),lr=0.1*lr,weight_decay=Config.weight_decay)


    # 训练
    for epoch in range(Config.max_epoch):

        for i ,(images,_) in enumerate(train_dataloader):
            if Config.use_gpu:
                images = images.cuda()
            images = images.view(-1, 28 * 28)

            if i%(Config.k+1)==Config.k:
                # 该loop对生成器的参数进行更新
                optimizer_g.zero_grad()
                z = torch.randn(images.shape[0],generator.fc1.in_features).to(Config.device)
                generative_images = generator(z)
                loss_g = G_loss(generative_images,discriminator)
                loss_g.backward()
                optimizer_g.step()
            else:
                # 该loop对判别器的参数进行更新
                optimizer_d.zero_grad()
                z = torch.randn(images.shape[0], generator.fc1.in_features).to(Config.device)
                generative_images = generator(z)
                loss_d = D_loss(images,generative_images,discriminator)
                loss_d.backward()
                optimizer_d.step()


            if i%Config.print_freq==Config.print_freq-1:
                # 当达到指定频率时，显示损失函数并画图
                print('Epoch:',epoch+1,'Round:',i+1,'Loss_g:',loss_g.item(),'Loss_d:',loss_d.item())


        GenerativePlot(generator, Config,random=False)

    generator.save()






train(Config)