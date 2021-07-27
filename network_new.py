import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
class generator(nn.Module):
    # initializers
    def __init__(self, d=64,input_nc=3,output_nc=3):
        super(generator, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc, d, 4, 2, 1)#input 3channels，output d channels,kernel size=4,stride=2,padding=1
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        #self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        #self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        #self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        #self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        #self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        #self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # #self.conv8_bn = nn.BatchNorm2d(d * 8)#原本就没有该BN层

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        #self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        #self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        #self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        #self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        #self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        #self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        #self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, output_nc, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        # e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        # e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        # e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        # e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        # e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        # e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e2 = self.conv2(F.leaky_relu(e1, 0.2))
        e3 = self.conv3(F.leaky_relu(e2, 0.2))
        e4 =self.conv4(F.leaky_relu(e3, 0.2))
        e5 = self.conv5(F.leaky_relu(e4, 0.2))
        e6 = self.conv6(F.leaky_relu(e5, 0.2))
        e7 = self.conv7(F.leaky_relu(e6, 0.2))

        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        # d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        # d1 = torch.cat([d1, e7], 1)
        # d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        # d2 = torch.cat([d2, e6], 1)
        # d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        # d3 = torch.cat([d3, e5], 1)
        # d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)#本来没有这句
        # d4 = torch.cat([d4, e4], 1)
        # d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        # d5 = torch.cat([d5, e3], 1)
        # d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        # d6 = torch.cat([d6, e2], 1)
        # d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))

        d1 = F.dropout(self.deconv1(F.relu(e8)), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2(F.relu(d1)), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3(F.relu(d2)), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4(F.relu(d3))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)#本来没有这句
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5(F.relu(d4))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6(F.relu(d5))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7(F.relu(d6))

        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64, input_nc=6):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(input_nc, d, 3, 1, 1)#判别器输入6 channels, 256*256*d
        self.conv1_2 = nn.Conv2d(d, d * 2, 3, 1, 1)  # 判别器输入6 channels, 256*256*2d
        self.conv2_1 = nn.Conv2d(d * 2, d * 2, 4, 2, 1)#128*128*2d
        self.conv2_2 = nn.Conv2d(d * 2, d * 4, 3, 1, 1)#128*128*4d
        self.conv2_bn = nn.BatchNorm2d(d * 4)
        self.conv3_1 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)#64*64*4d
        self.conv3_2 = nn.Conv2d(d * 4, d * 8, 3, 1, 1)#64*64*8d
        self.conv3_bn = nn.BatchNorm2d(d * 8)
        self.conv4 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)# 32*32*8d
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)
        # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1_2(self.conv1_1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2_2(self.conv2_1(x))), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3_2(self.conv3_1(x))), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x #输出是0-1之间的数[31*31]，表示判成真实的概率
# class spe_ran_discriminator(nn.Module):
#     def __init__(self,input_nc=150,number=16):
#         super(spe_ran_discriminator, self).__init__()
#         self.number=number
#         self.fc_1=nn.Linear(input_nc,128)
#         self.fc_2=nn.Linear(128,256)
#         self.fc_3=nn.Linear(256,128)
#         self.fc_4=nn.Linear(128,2)
#     def forward(self,input_real,result):# 150 256 256
#         h=input_real.shape[2]
#         w=input_real.shape[3]
#         numbers = int(math.sqrt(self.number))#M=16,化块小了
#         real = torch.zeros(numbers, numbers).cuda()
#         pre = torch.zeros(numbers, numbers).cuda()
#         for save_h in range(numbers):
#             for save_w in range(numbers):
#                 location=random.randrange(h*w)
#                 loc_h=math.floor(location/w)
#                 loc_w=location-loc_h*w
#                 real_spectral=input_real[:,:,loc_h,loc_w]#1*150不行，要再扩一维
#                 pre_spectral=result[:,:,loc_h,loc_w]
#                 real[save_h,save_w]= torch.sigmoid(self.fc_4(self.fc_3(self.fc_2(self.fc_1(torch.unsqueeze(real_spectral,1))))))[0,0,0]
#                 pre [save_h,save_w] = torch.sigmoid(self.fc_4(self.fc_3(self.fc_2(self.fc_1(torch.unsqueeze(pre_spectral,1))))))[0,0,0]
#         return real,pre
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)

class spe_discriminator(nn.Module):
    def __init__(self,input_nc=150,inter=64):
        super(spe_discriminator, self).__init__()
        self.inter=inter
        self.fc_1=nn.Linear(input_nc,128)
        self.fc_2=nn.Linear(128,256)
        self.fc_3=nn.Linear(256,128)
        self.fc_4=nn.Linear(128,2)
    def forward(self,input_real,result):# 150 256 256
        location_h = random.randrange(self.inter)
        location_w = random.randrange(self.inter)
        numbers=int(math.ceil(input_real.shape[2]/(self.inter)))
        real=torch.zeros(numbers,numbers).cuda()
        pre =torch.zeros(numbers,numbers).cuda()
        for h in range(numbers):
            for w in range(numbers):
                loc_h=location_h+self.inter*h
                loc_w=location_w+self.inter*w
                real_spectral=input_real[:,:,loc_h,loc_w]#1*150不行，要再扩一维
                pre_spectral=result[:,:,loc_h,loc_w]
                real[h, w] = torch.sigmoid(self.fc_4(self.fc_3(self.fc_2(self.fc_1(torch.unsqueeze(real_spectral, 1))))))[0, 0, 0]
                pre[h, w] = torch.sigmoid(self.fc_4(self.fc_3(self.fc_2(self.fc_1(torch.unsqueeze(pre_spectral, 1))))))[0, 0, 0]
                # real[h,w]= torch.sigmoid(self.fc_4(F.relu(self.fc_3(F.relu(self.fc_2(F.relu(self.fc_1(torch.unsqueeze(real_spectral,1)))))))))[0,0,0]
                # pre [h,w] = torch.sigmoid(self.fc_4(F.relu(self.fc_3(F.relu(self.fc_2(F.relu(self.fc_1(torch.unsqueeze(pre_spectral,1)))))))))[0,0,0]
        return real,pre
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()