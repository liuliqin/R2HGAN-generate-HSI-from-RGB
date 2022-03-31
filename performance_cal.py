import os
import gdal
import argparse
import glob
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import math


from PIL import  Image
def read_tif(filename):
    dataset = gdal.Open(filename)
    if dataset is None:
        print("文件%s无法打开" % filename)
        exit(-1)
    im_data = dataset.ReadAsArray()
    return im_data

def ssim(img1, img2):
    C1 = (0.01 * 65535)**2
    C2 = (0.03 * 65535)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        ssims = []
        for i in range(img1.shape[0]):
           ssims.append(ssim(img1[i,:,:], img2[i,:,:]))#改
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')



def cal_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(65535.0 / math.sqrt(mse))

def calculate_psnr(img1, img2):
    """calculate psnr among multi-channel
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return psnr(img1, img2)
    elif img1.ndim == 3:
        sum_psnr = 0
        for i in range(img1.shape[0]):
            this_psnr = cal_psnr(img1[i,:,:], img2[i,:,:])
            sum_psnr += this_psnr
    return sum_psnr/img1.shape[0]



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=False, default='g5pair', help='')
    # parser.add_argument('--model_path', required=False, default='20210319 100g_BNo', help='model path')
    # parser.add_argument('--model_epoch', default='800', help='Which model to test?')
    # parser.add_argument('--test_flag', default='test', help='test on train or test')
    parser.add_argument('--cal_path', default='20220315_aviris/test_results_50/test', help='cal path')
    parser.add_argument('--data_path',default='data\mini_aviris_2202/test')
    opt = parser.parse_args()
    print(opt)

    data_path= opt.data_path
    real_path=os.path.join(data_path,'*.tif')#找到原图
    real_img_name=glob.glob(real_path)
    all_rmse=0
    all_psnr=0
    all_mrae=0
    all_sam=0
    all_mssim=0
    all_psnr=0
    f=open(opt.cal_path+'/performance.txt','w')
    n=0
    for img_name in real_img_name:
        n+=1
        print('%d image calculating' % n)
        f.write(img_name)
        label = read_tif(img_name).astype('float64')#n*h*w
        result_path = opt.cal_path+'/'+img_name.split('\\')[-1]
        result = read_tif(result_path).astype('float64')
        channel,height,width=result.shape

        # cal RMSE，MRAE,SAM
        sum_se=0
        sum_mrae=0
        sum_sam=0

        for i in range(0,height):
           for j in range(0,width):
              sum_se += np.sum((result[:,i,j]-label[:,i,j])**2)
              A=label[:, i, j]
              sum_mrae += np.sum(abs(result[:,i,j]-label[:,i,j])/(label[:,i,j]+1))
              spe_res=result[:,i,j].reshape(1,-1)
              spe_lab=label[:,i,j].reshape(1,-1)
              sum_sam +=math.acos(cosine_similarity(spe_lab,spe_res))

        rmse=(sum_se/(height*width*channel))**0.5
        mrae=sum_mrae/(height*width*channel)
        sam=sum_sam/(height*width)      # sam=0

        # cal mssim psnr
        mssim=calculate_ssim(result,label)
        psnr = calculate_psnr(result, label)
        f.write('   rmse='+str(rmse)+'   mrae='+str(mrae)+'   sam='+str(sam)+'  mssim='+str(mssim) + '   mpsnr='+str(psnr))
        f.write('\n')
        all_rmse +=rmse
        all_psnr +=psnr
        all_mrae +=mrae
        all_sam +=sam
        all_mssim +=mssim
    mean_rmse=all_rmse/len(real_img_name)
    mean_psnr = all_psnr / len(real_img_name)
    mean_rae = all_mrae / len(real_img_name)
    mean_sam = all_sam / len(real_img_name)
    mean_ssim = all_mssim / len(real_img_name)
    f.write('图像平均 \n  rmse='+str(mean_rmse)+'  mrae='+str(mean_rae)+'  sam='+str(mean_sam)+'  ssim='+str(mean_ssim)+'  psnr='+str(mean_psnr))
    f.write('\n')
    f.close()


