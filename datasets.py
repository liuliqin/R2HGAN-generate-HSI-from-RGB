
import os
import gdal
import random

from torch.utils.data import Dataset
from PIL import Image

import cv2
import torch
from torchvision import transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, folder,crop_size):
        self.folder = folder
        self.multi_path = os.listdir(folder + '/multi')
        self.crop_size=crop_size

    def __len__(self):
        return len(self.multi_path)

    def __getitem__(self, index):

        multi_img=cv2.imread(self.folder+'/multi/'+self.multi_path[index]).astype('float32')

        # multi_nc=multi_img.shape[2]#获取通道数
        hs,ws,h,w=self.randomcrop(multi_img,self.crop_size)
        multi_imgc=torch.tensor(multi_img[hs:hs+h,ws:ws+w,:].transpose(2,0,1)/255)
        # normalize
        # mean_multi=tuple(multi_nc*[self.mean])
        # std_multi = tuple(multi_nc * [self.std])
        #transform_m = transforms.Compose([#transforms.ToTensor(),
                                       #   transforms.Normalize(mean=mean_multi, std=std_multi)])
        #multi_imgt=transform_m(multi_imgc)
        #高光谱
        hyper_path = self.folder+'/hyper/'+self.multi_path[index]

        dataset = gdal.Open(hyper_path)
        if dataset is None:
            print("文件%s无法打开" % hyper_path)
            exit(-1)
        im_data = dataset.ReadAsArray()
        hyper = im_data[:, hs:hs+h,ws:ws+w]
        # hyper_img=hyper.transpose(1, 2, 0)
        hyper_img = torch.tensor(hyper.astype('float32')/4095)

        #transform
        # im_bands = dataset.RasterCount-2#后两维不是高光谱
        # mean_hyper=tuple(im_bands*[self.mean])
        # std_hyper=tuple(im_bands*[self.std])

        #transform_h = transforms.Compose([transforms.Normalize(mean=mean_hyper, std=std_hyper)]) # normalize:
        # hyper_imgt=transform_h(hyper_img)
        return multi_imgc, hyper_img

    def randomcrop(self,img,crop_size):
        h,w,c= img.shape
        th= tw = crop_size
        if w == tw and h == th:
            return 0,0,h,w
        if w < tw or h < th:
            print('图片大小未达到裁切尺寸')
            exit(-1)
        i = random.randint(0, h - th-10)
        j = random.randint(0, w - tw-10)
        return i,j,th,tw






