import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#卷积缩小2倍 #此法是图片色彩变换
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#逆卷积放大2倍 #此法是图片色彩变换
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


#G 3*576*720--->3*576*720
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.samll = nn.Sequential(
            nn.Conv2d(3, 3, 4, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),
            
            nn.Conv2d(3, 3, 4, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),

            nn.Conv2d(3, 1, 4, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),
        )

        self.large = nn.Sequential(
            nn.ConvTranspose2d(1, 3, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),

            nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),

            nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU(True),
        )

        self.fc=nn.Sequential(
            nn.Linear(1*72*90, 1*72*90),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.samll(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, 72, 90)
        x = self.large(x)
        return x

#D  3*576*720--->3*576*720
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 3, 4, stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.Conv2d(3, 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(2),
            nn.Conv2d(2, 1, 4, stride=2, padding=1),
            nn.InstanceNorm2d(1),
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)