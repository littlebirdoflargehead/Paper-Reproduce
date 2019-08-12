import torch
import models
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from utils import VAE_Loss,VampPriorVAE_Loss,ImageVsReImagePlot,GenerativePlot
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
    model = getattr(models, Config.model)()
    if Config.load_model_path:
        model.load(Config.load_model_path)
    if Config.use_gpu:
        model.to(Config.device)

    # step3: 目标函数与优化器

    lr = Config.lr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=Config.weight_decay)


    # 训练
    for epoch in range(Config.max_epoch):

        for i ,(images,_) in enumerate(train_dataloader):
            if Config.use_gpu:
                images = images.cuda()
            images = images.view(-1, 28 * 28)

            optimizer.zero_grad()
            loss = VampPriorVAE_Loss(images, model)
            re_images, _, _ = model(images)
            # re_images,mu,logvar = model(images)
            # loss = VAE_Loss(images,re_images,mu,logvar)

            loss.backward()
            optimizer.step()


            if i%Config.print_freq==Config.print_freq-1:
                # 当达到指定频率时，显示损失函数并画图
                print('Epoch:',epoch+1,'Round:',i+1,'Loss:',loss.item())
                ImageVsReImagePlot(images,re_images,Config)


        model.save()
        GenerativePlot(model, Config,random=True)


def Marginal_Likelihood_Evaluate(model,Config):

    # step1: 数据
    test_dataset = torchvision.datasets.MNIST(root=Config.test_data_root,train=False,transform=transforms.ToTensor(),download=False)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=10000,shuffle=False,num_workers=Config.num_workers)

    # step: 抽样计算边缘
    L = 5
    LikeLihood = 0
    for data,_ in iter(test_dataloader):
        if Config.use_gpu:
            images = data.cuda()
        images = images.view(-1, 28 * 28)
        mu,logvar = model.encoder(images)
        loss = 0
        for l in range(L):
            z,epslon = model.reparameter_trick(mu,logvar)
            re_images = model.decoder(z)

            BCE = torch.sum(torch.log(re_images)*images+torch.log(1-re_images)*(1-images),dim=1)
            KLD = 0.5 *torch.sum(torch.pow(epslon, 2) + logvar - torch.pow(z, 2),dim=1)
            loss = loss + torch.exp(BCE+KLD)
        loss = torch.sum(torch.log(loss/L))
        LikeLihood = LikeLihood+loss
    return LikeLihood







train(Config)