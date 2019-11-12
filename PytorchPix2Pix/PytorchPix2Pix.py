from dataloader import TrainPhotos,TestPhotos
from Net import GeneratorUNet,Discriminator,weights_init_normal
from torchvision.utils import save_image
from torch.autograd import Variable

import torch
import torch.nn as nn
import numpy as np

#加载图片
TrainData=TrainPhotos('./train')
TestData=TestPhotos('./test')

#加载网络
generator = GeneratorUNet()
discriminator = Discriminator()

#优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

#损失函数
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
lambda_pixel = 100

#网络权重初始化
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


if __name__ == '__main__':
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    num = 0

    for k in range(500):
        for i in range(4):
            testdata = TestData[i].cuda()
            traindata = TrainData[i].cuda()
            testdata = testdata.unsqueeze(0).cuda()
            traindata = traindata.unsqueeze(0).cuda()
            real_A = traindata.cuda()
            real_B = testdata.cuda()

            valid = Variable(torch.Tensor(np.ones((real_A.size(0), 1, 72, 90))), requires_grad=False).cuda()
            fake = Variable(torch.Tensor(np.zeros((real_A.size(0), 1, 72, 90))), requires_grad=False).cuda()

            #train G
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            #train D
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            #save fake photo
            save_image(fake_B, './dc_img/fake_images-{}.png'.format(num))
            num = num+1
            print(num)

#save G and D
torch.save(generator,'NetG.pkl')                             #entire net
torch.save(generator.state_dict,'NetG_parames.pkl')          #parameters
torch.save(discriminator,'NetD.pkl')                         #entire net
torch.save(discriminator.state_dict,'NetD_parames.pkl')      #parameters