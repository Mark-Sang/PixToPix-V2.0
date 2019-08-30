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

#网络权重初始化
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if __name__ == '__main__':
    traindata = TrainData[3].cuda()
    traindata = traindata.unsqueeze(0).cuda()
    real_A = traindata.cuda()
    netG = torch.load('NetG.pkl')
 #   generator.load_state_dict(torch.load('NetG_parames.pkl'))
    fake_B = netG(real_A)
    save_image(fake_B, './dc_img/fake_images-{}.png'.format(12))