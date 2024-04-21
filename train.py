import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model import Back_PVT
from data import get_loader
from dataloader import test_dataset
from utils import clip_gradient,adjust_lr
import os
from scipy import misc
import smoothness
import imageio
import logging
from tensorboardX import SummaryWriter
from loss import *
import warnings
warnings.filterwarnings("ignore")

def load_matched_state_dict(model, state_dict, print_stats=True):
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def visualize_prediction1(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal1.png'.format(kk)
        imageio.imsave(save_path + name, pred_edge_kk)

def visualize_prediction2(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_sal2.png'.format(kk)
        imageio.imsave(save_path + name, pred_edge_kk)

def val(model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        test_loader = test_dataset(image_root=opt.test_path + '/image/',
                            gt_root=opt.test_path + '/mask/',
                            testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res0,res1,res,_,_= model(image)
            # res0,res1,res= model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # imageio.imsave(save_path+name, res)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

def visualize_edge(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_edge.png'.format(kk)
        imageio.imsave(save_path + name, pred_edge_kk)

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, masks, grays, edges = pack
        images = Variable(images)
        gts = Variable(gts)
        masks = Variable(masks)
        grays = Variable(grays)
        edges = Variable(edges)
        images = images.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        edges = edges.cuda()

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        criterion = [SimMaxLoss(metric='cos', alpha=0.25).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=0.25).cuda()]

        sal1, edge_map, sal2, fg_feats, bg_feats = model(images)

        loss1 = criterion[0](fg_feats)
        loss2 = criterion[1](bg_feats, fg_feats)
        loss3 = criterion[2](bg_feats)

        sal1_prob = torch.sigmoid(sal1)
        sal1_prob = sal1_prob * masks
        sal2_prob = torch.sigmoid(sal2)
        sal2_prob = sal2_prob * masks

        smoothLoss_cur1 = opt.sm_loss_weight*smooth_loss(torch.sigmoid(sal1), grays)
        sal_loss1 = ratio*CE(sal1_prob, gts*masks)+smoothLoss_cur1
        smoothLoss_cur2 = opt.sm_loss_weight * smooth_loss(torch.sigmoid(sal2), grays)
        sal_loss2 = ratio * CE(sal2_prob, gts * masks) + smoothLoss_cur2
        edge_loss = opt.edge_loss_weight*CE(torch.sigmoid(edge_map),edges)
        bce = sal_loss1+edge_loss+sal_loss2
        visualize_prediction1(torch.sigmoid(sal1))
        visualize_edge(torch.sigmoid(edge_map))
        visualize_prediction2(torch.sigmoid(sal2))

        loss = bce + loss1 + loss2 +loss3  
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data))
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], sal1_loss: {:0.4f}, edge_loss: {:0.4f}, sal2_loss: {:0.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step, sal_loss1.data, edge_loss.data, sal_loss2.data))

    # if epoch % 10 == 0:
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), opt.save_path + 'scribble' + '_%d'  % epoch  + '.pth')

if __name__ == '__main__':

    model_name = 'xxx' 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
    parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
    parser.add_argument('--edge_loss_weight', type=float, default=1.0, help='weight for edge loss')
    parser.add_argument('--test_path', type=str,default='xxx',help='path to testing dataset')
    parser.add_argument('--gpu_devices', type=str, default='0,3', help='GPU devices to use for training')
    parser.add_argument('--save_path', type=str,default='xxx/'+model_name+'/')
    parser.add_argument('--epoch_save', type=int,default=1, help='every n epochs to save model')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("Network-Train")

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_devices

    print('Learning Rate: {}'.format(opt.lr))
    # build models
    model = Back_PVT(channel=32)
    model = DataParallel(model)

    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Sucefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)


    # image_root = 'XXXX/data/MSD_scribble/train/image/'
    # gt_root = 'XXXX/data/MSD_scribble/train/gt/'
    # mask_root = 'XXXX/data/MSD_scribble/train/mask/'
    # edge_root = 'XXXX/data/MSD_scribble/train/edge/'
    # grayimg_root = 'XXXX/data/MSD_scribble/train/gray/'

    image_root = 'XXXX/data/PMD_scribble/train/image/'
    gt_root = 'XXXX/data/PMD_scribble/train/gt/'
    mask_root = 'XXXX/data/PMD_scribble/train/mask/'
    edge_root = 'XXXX/data/PMD_scribble/train/edge/'
    grayimg_root = 'XXXX/data/PMD_scribble/train/gray/'

    # image_root = 'XXXX/data/RGBD_Mirror_scribble/train/image/'
    # gt_root = 'XXXX/data/RGBD_Mirror_scribble/train/gt/'
    # mask_root = 'XXXX/data/RGBD_Mirror_scribble/train/mask/'
    # edge_root = 'XXXX/data/RGBD_Mirror_scribble/train/edge/'
    # grayimg_root = 'XXXX/data/RGBD_Mirror_scribble/train/gray/'

    train_loader = get_loader(image_root, gt_root, mask_root, grayimg_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    CE = torch.nn.BCELoss()
    smooth_loss = smoothness.smoothness_loss(size_average=True)

    writer = SummaryWriter(opt.save_path + 'summary')

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0
    for epoch in range(1, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        if epoch % opt.epoch_save==0:
            val(model, epoch, opt.save_path, writer)
