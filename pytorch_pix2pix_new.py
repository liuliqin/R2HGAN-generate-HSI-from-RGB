import os, time, pickle, argparse, network_new,random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from datasets import ImageDataset
from util import Logger
#import numpy as np

parser = argparse.ArgumentParser()#创建对象
parser.add_argument('--dataset', required=False, default='g5pair',  help='')#添加参数
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=256, help='crop size')

parser.add_argument('--train_epoch', type=int, default=701, help='number of train epochs')
parser.add_argument('--G_only_epoch', type=int, default=100, help=' train G only epochs')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')#调学习率，保证初始是 0.0001
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--G_Decay_epoch', type=float, default=300, help='learning rate decay epoch, default=100')
parser.add_argument('--D_Decay_epoch', type=float, default=300, help='learning rate decay epoch, default=100')
parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
# parser.add_argument('--Cos_lambda', type=float, default=0, help='lambda for cosine loss')
parser.add_argument('--spe_lambda', type=float, default=1, help='lambda for cosine loss')
# parser.add_argument('--D_spe_lambda', type=float, default=1, help='lambda for cosine loss')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')#0.5
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--work_root', required=False, default='20210528 m64', help='results save path')
opt = parser.parse_args()#解析参数
print(opt)#显示参数

# 创建结果保存路径
root = opt.dataset + '_' + opt.save_root + '/'+opt.work_root+'/'#mul2hyper_results/
model_root=root + 'model/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(model_root):
    os.mkdir(model_root)
#判断是否存在初始模型
G_dir=os.path.join(model_root,'g5pair_generator_param_400.pth')
D_spa_dir=os.path.join(model_root,'g5pair_discriminator_spa_param_400.pth')
D_spe_dir=os.path.join(model_root,'g5pair_discriminator_spe_param_400.pth')

train_dataset = ImageDataset('./data/'+opt.dataset+'/train',opt.crop_size)
test_dataset = ImageDataset('./data/'+opt.dataset+'/test',opt.crop_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
fixed_multi, fixed_hyper = train_loader.__iter__().__next__()
multi_channel=fixed_multi.size()[1]
hyper_channel=fixed_hyper.size()[1]
input_nc=multi_channel+hyper_channel
# network
G = network_new.generator(opt.ngf, multi_channel,hyper_channel)#生成器是一个编解码Unet，kernel_d=ngf

D_spa = network_new.discriminator(opt.ndf, input_nc)
D_spe =network_new.spe_discriminator(hyper_channel)

G.weight_init(mean=0.0, std=0.02)
D_spa.weight_init(mean=0.0, std=0.02)
D_spe.weight_init(mean=0.0, std=0.02)
G.cuda()
D_spa.cuda()
D_spe.cuda()

G.train()
D_spa.train()
D_spe.train()

# loss
BCE_loss = nn.BCELoss().cuda()#判别是真是假改成MSE更好？
L1_loss = nn.L1Loss().cuda()
Cos_loss= nn.CosineSimilarity(dim=1).cuda()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_spa_optimizer = optim.Adam(D_spa.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
D_spe_optimizer = optim.Adam(D_spe.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))


G_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=opt.G_Decay_epoch, gamma=0.1)
D_spa_scheduler = torch.optim.lr_scheduler.StepLR(D_spa_optimizer, step_size=opt.D_Decay_epoch, gamma=0.1)
D_spe_scheduler = torch.optim.lr_scheduler.StepLR(D_spe_optimizer, step_size=opt.D_Decay_epoch, gamma=0.1)


if os.path.exists(G_dir) and os.path.exists(D_spa_dir) and os.path.exists(D_spe_dir):
    G.load_state_dict(torch.load(G_dir))
    D_spa.load_state_dict(torch.load(D_spa_dir))
    D_spe.load_state_dict(torch.load(D_spe_dir))
    start_epoch =400
    print('加载成功！')
else:
    start_epoch = 0
    print('无保存模型，将从头开始训练！')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
logger = Logger(opt.train_epoch, len(train_loader))
num_iters=0
for epoch in range(start_epoch+1,opt.train_epoch):
    # D_spa_losses = []
    # D_spe_losses=[]
    # G_losses = []
    epoch_start_time = time.time()
    num_iter = 0

    for x_, y_ in train_loader:
        x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
        # train discriminator D
        D_spa.zero_grad()  # 把判别器的梯度置0
        D_spe.zero_grad()  # 把判别器spe的梯度置0

        D_result = D_spa(x_, y_).squeeze()
        D_real_loss = BCE_loss(D_result,
                               Variable(torch.ones(D_result.size()).cuda()))  # 真实数据的loss，希望判定真实数据为真实的概率是1，因此概率与1比
        G_result = G(x_)
        D_result = D_spa(x_, G_result).squeeze()
        D_fake_loss = BCE_loss(D_result,
                               Variable(torch.zeros(D_result.size()).cuda()))  # 生成数据的loss，希望判定生成数据的真实概率为0，因此和0比

        D_spa_loss = (D_real_loss + D_fake_loss) * 0.5
        pre_real, pre_result = D_spe(y_, G_result)
        D_spe_loss = (BCE_loss(pre_real, Variable(torch.ones(pre_real.size()).cuda())) + BCE_loss(pre_result, Variable(
            torch.zeros(pre_real.size()).cuda())))*0.5
        D_loss = D_spa_loss + D_spe_loss
        #训练D的条件
        if (num_iters % 3 == 0) and (epoch>opt.G_only_epoch):
        # if epoch>opt.G_only_epoch:
            D_loss.backward(retain_graph=True)#loss反传
            D_spa_optimizer.step()#优化
            D_spe_optimizer.step()# 优化
        # train_hist['D_losses'].append(D_spa_loss.item())#D_losses加入当前数值
        # train_hist['D_losses'].append(D_spa_loss.item())
        # D_spa_losses.append(D_spa_loss.item())

        # train generator G
        G.zero_grad()#生成器梯度置0
        pre_real, pre_result = D_spe(y_, G_result)
        G_L1_loss = L1_loss(G_result, y_)  # 希望生成数据和y尽可能接近
        D_result = D_spa(x_, G_result).squeeze()
        G_BCE_loss= BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))
        G_spe_loss=BCE_loss(pre_result,Variable(torch.ones(pre_result.size()).cuda()))
        # G_cos_loss = 1 - torch.mean(Cos_loss(G_result, y_))
        G_train_loss = G_BCE_loss + opt.spe_lambda * G_spe_loss + opt.L1_lambda * G_L1_loss
                       # + opt.Cos_lambda * G_cos_loss #加入余弦相似度loss
        G_train_loss.backward()
        G_optimizer.step()

        # train_hist['G_losses'].append(G_train_loss.item())

        # G_losses.append(G_train_loss.item())

        num_iter += 1
        num_iters += 1

        logger.log({'D_real_loss': D_real_loss,'D_fake_loss': D_fake_loss, 'D_spa_loss': D_spa_loss,
                    'D_spe_loss': D_spe_loss,'D_loss': D_loss,'G_BCE_loss': G_BCE_loss,
                    'G_L1_loss': G_L1_loss,'G_spe_loss': G_spe_loss,'G_train_loss': G_train_loss},#'G_cos_loss': G_cos_loss,
                   images={'MSI': x_, 'real_HSI': y_, 'fake_HSI': G_result})

    D_spe_scheduler.step()
    D_spa_scheduler.step()
    G_scheduler.step()
    if (epoch%50 ==0): #and (epoch>0):#每50步保存一个模型

        torch.save(G.state_dict(), model_root+ model + 'generator_param_'+str(epoch)+'.pth')
        torch.save(D_spa.state_dict(), model_root+ model + 'discriminator_spa_param_'+str(epoch)+'.pth')
        torch.save(D_spe.state_dict(), model_root + model + 'discriminator_spe_param_' + str(epoch) + '.pth')
