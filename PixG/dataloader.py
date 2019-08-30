import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1] 
])

#定义自己的train数据集合
class TrainPhotos(data.Dataset):
    def __init__(self,root):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        imgs.sort(key=lambda x:int(x[:-4]))      #图片名从小到大排序
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

#定义自己的test数据集合
class TestPhotos(data.Dataset):
    def __init__(self,root):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        imgs.sort(key=lambda x:int(x[:-4]))      #图片名从小到大排序
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)

