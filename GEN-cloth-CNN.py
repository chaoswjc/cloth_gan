##GAN
#生成对抗网络(Generative Adversarial Networks, GANs)、
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
from torchvision import  transforms
from torchvision.transforms import ToPILImage,ToTensor
import  numpy as np


from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils import data
import os


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 100,100)
    return out

to_pil=ToPILImage()
batch_size = 4

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# mnist = datasets.MNIST('./data', transform=img_transform)

class Clothes(data.Dataset):
    def __init__(self,root,transforms=None):
        imgs=os.listdir(root)
        #生成所有文件名字的列表
        # print(imgs[1])   cat.1.jpg
        ##不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
        self.imgs=[os.path.join(root,img) for img in imgs]
        self.tranfroms=transforms
    def __getitem__(self,index):
        img_path=self.imgs[index]
        #dog 1    cat 0
        # print(img_path.split('/'))
        label = 1  #全部设置成1
        data=Image.open(img_path)

        # data=to_tensor(pil_img)  #这个结果是 3 280 500
        # array=np.asarray(pil_img) #这个结果是  280 500 3  差不多
        # data=torch.from_numpy(array)  #两句应该等价于 to_tensor(pil_img)

        if self.tranfroms:
            data=self.tranfroms(data)
        return data ,label
    def __len__(self):
        return len(self.imgs)

#250 130
#dataset=Clothes('d:/pytorch/imgs/uniform/each100/',img_transform)
dataset=Clothes('/home/jcwang/cloth/each100/',img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#3 100 100

#import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import os


num_epoch = 500
z_dimension = 100


#4 3 100 100
# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2,stride=2),  # batch, 32, 50,50
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 32, 25,25
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 25, 25
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 12,12
        )
        self.fc = nn.Sequential(
            nn.Linear(64*12*12, 1024),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
#
# class discriminator(nn.Module):
#     def __init__(self):
#         super(discriminator, self).__init__()
#         self.dis = nn.Sequential(
#             nn.Linear(7500, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())
#
#     def forward(self, x):
#         x = self.dis(x)
#         return x


# Generator
class generator(nn.Module):
    def __init__(self, input_size, num_feature):

        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 10000=1x100x100
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 100, 100
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 100, 100
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 3, 2, stride=2),  # batch, 3, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        #print(x.size())
        x = x.view(x.size(0), 1, 200, 200)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

# class generator(nn.Module):
#     def __init__(self):
#         super(generator, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(100, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 7500), nn.Tanh())
#
#     def forward(self, x):
#         x = self.gen(x)
#         return x


D = discriminator()
D=D.cuda()
#G = generator()
G = generator(z_dimension, 40000)
G=G.cuda()
# Binary cross entropy loss and optimizer
criterion = nn.BCELoss().cuda()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# Start training
four=0;
all_4=[1,2,3,4];
for epoch in range(num_epoch):
    print(epoch)
    for i, (img, _) in enumerate(dataloader):
        # print(img.size())   #4 3 50 50
        num_img = img.size(0) #4
        # =================train discriminator
        #img = img.view(num_img, -1)   #8 750
        real_img = Variable(img).cuda() #4 3  28  28
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # compute loss of real_img
        real_out = D(real_img)
        real_out.data=real_out.data.view(-1)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        #print(fake_img.size())
        #print(fake_img.size())
        #print(fake_img.size())
        fake_out = D(fake_img)
        fake_out.data = fake_out.data.view(-1)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        output.data = output.data.view(-1)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        #if (i + 1) % 20 == 0:
        #if epoch == 0:
            #real_images = to_img(real_img.data)
            #save_image(real_images, 'd:/pytorch/imgs/uniform/new/real/real_imgs'+str(epoch)+'_'+str(i)+'.png')
            #save_image(real_images, '/home/jcwang/cloth/real/real_imgs' + str(epoch) + '_' + str(i) + '.png')
        if fake_scores.data.mean()>=0.2:
            fake_images = to_img(fake_img.data)
            all_4[four]=fake_images
            four+=1
            if four==4:
                fake_images = torch.cat([all_4[0], all_4[1], all_4[2], all_4[3]], 0)
                four=0;
                fake_images=tv.utils.make_grid(fake_images,4)
                save_image(fake_images, '/home/jcwang/cloth/fakeCNN/fake_imgs'+str(epoch)+'_'+str(i)+'.png')
    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
          'D real: {:.6f}, D fake: {:.6f}'.format(
        epoch, num_epoch, d_loss.data[0], g_loss.data[0],
        real_scores.data.mean(), fake_scores.data.mean()))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')