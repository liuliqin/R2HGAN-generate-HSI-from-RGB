import itertools, imageio, torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from scipy.misc import imresize
from visdom import Visdom
import sys
import time
import datetime

def tensor2image(tensor):
    imtensor=tensor[0]
    if imtensor.size()[0]== 3:
        imbinar = imtensor
    elif imtensor.size()[0]<3:
        imbinar=np.tile(imtensor[0],(3,1,1))
    else:
        imbinar = imtensor[[3, 7, 10], :, :]
    image = 127.5 * (imbinar.cpu().float().numpy() + 1.0)
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            # print(tensor.requires_grad)
            if image_name not in self.image_windows:

                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.detach()), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.detach()), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1



#def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    # test_images = G(x_)
    #
    # size_figure_grid = 3
    # fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    # for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    #
    # for i in range(x_.size()[0]):
    #     ax[i, 0].cla()
    #     ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    #     ax[i, 1].cla()
    #     ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    #     ax[i, 2].cla()
    #     ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)
    #
    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, 0.04, label, ha='center')
    #
    # if save:
    #     plt.savefig(path)
    #
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

