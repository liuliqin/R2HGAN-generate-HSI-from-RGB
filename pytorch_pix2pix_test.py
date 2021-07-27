import torch, network_new, argparse, os
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import util
from datasets import ImageDataset
import numpy as np
import gdal

def write_img(filename, im_data):#im_geotrans, im_proj,
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    #dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    #dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='g5pair',  help='')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=512, help='input size')
    parser.add_argument('--model_path',required=False, default='20210528 m64',help='model path')
    parser.add_argument('--model_epoch', default='600', help='Which model to test?')
    parser.add_argument('--test_flag', default='test',help='test on train or test')
    opt = parser.parse_args()
    print(opt)

    test_dataset = ImageDataset('./data/'+opt.dataset+'/'+opt.test_flag ,opt.input_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    save_root= opt.dataset + '_results/'+ opt.model_path + '/test_results_'+ opt.model_epoch

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    save_dir= save_root+'/'+opt.test_flag
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    fixed_multi, fixed_hyper = test_loader.__iter__().__next__()
    multi_channel=fixed_multi.size()[1]
    hyper_channel=fixed_hyper.size()[1]
    input_nc=multi_channel+hyper_channel
    G = network_new.generator(opt.ngf,multi_channel,hyper_channel)
    G.cuda()
    G_path=opt.dataset + '_results/' + opt.model_path+'/model/'+opt.dataset + '_generator_param_'+opt.model_epoch +'.pth' # mul2hyper_results\20201111-moving BN\models
    checkpoint=torch.load(G_path)
    G.load_state_dict(checkpoint)

    # network
    n = 0
    print('test start!')
    for x_, y_ in test_loader:
        with torch.no_grad():
            x_ = x_.cuda()
        test_image = G(x_)
        s = test_loader.dataset.multi_path[n][0:-4]
        path = save_dir + '/' + s + '_input.png'
        plt.imsave(path, x_[0].cpu().detach().numpy().transpose(1, 2, 0))

        hyper_image = test_image[0].cpu().detach().numpy().transpose(1, 2, 0)
        path = save_dir + '/' + s + '_output.tif'
        hyper_img=(hyper_image * 4095).astype('uint16')
        hyper_data=hyper_img.transpose(2,0,1)#c*h*w
        write_img(path, hyper_data)

        y_image = y_[0].cpu().detach().numpy().transpose(1,2,0)
        save_path=save_dir + '/' + s + '_real.tif'
        y_img = (y_image * 4095).astype('uint16')
        y_data = y_img.transpose(2, 0, 1)  # c*h*w
        write_img(save_path, y_data)

        n += 1

        print('%d images generation complete!' % n)

