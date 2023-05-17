# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py
# cam实验，验证loss系数
import argparse
import os
import pdb
import random
import logging
import numpy as np
import time
import setproctitle

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from models.TransBTS.Layer_changed import TransBTS
# from models.TransBTS.FCN import TransBTS
import torch.distributed as dist
from models import criterions

from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn
from data.ADNI import ADNI
# from data.HarP import ADNI
# from M3d_Cam.cam import medcam
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, recall_score, precision_score
from medpy import metric

# from models.autoweighted_loss import AutomaticWeightedLoss
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='leizhenxin', type=str)
parser.add_argument('--experiment', default='uncertainty_loss_MCICN', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description',
                    default='HarP'
                            'training on train.txt!',
                    type=str)

# DataSet Information

parser.add_argument('--root', default='/hy-tmp/processed', type=str)
parser.add_argument('--train_dir', default='', type=str)
parser.add_argument('--valid_dir', default='', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train_data.txt', type=str)
parser.add_argument('--valid_file', default='test_data.txt', type=str)
parser.add_argument('--dataset', default='brats', type=str)
parser.add_argument('--model_name', default='TransBTS', type=str)
parser.add_argument('--input_C', default=1, type=int)
parser.add_argument('--input_H', default=256, type=int)
parser.add_argument('--input_W', default=256, type=int)
parser.add_argument('--input_D', default=156, type=int)
parser.add_argument('--crop_H', default=128, type=int)
parser.add_argument('--crop_W', default=128, type=int)
parser.add_argument('--crop_D', default=128, type=int)
parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=2e-5, type=float)  # 0.002
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='softmax_dice', type=str)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=600, type=int)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--test_freq', default=1, type=int)
parser.add_argument('--load', default=True, type=bool)
parser.add_argument('--cls_start', default=50, type=int)
parser.add_argument('--apply_uncertainty_loss', default=5, type=int)
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    # model = medcam.inject(model, backend='gcam', output_dir='', layer='heatmap_conv',
    #                       return_attention=True, retain_graph=True)

    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    # set higher lr for cls
    fc_params = []
    other_params = []
    for name, params in model.named_parameters():
        # print(name)
        if name in ['module.fc.weight']:
            # import pdb; pdb.set_trace()
            fc_params += [params]
        else:
            other_params += [params]
    # import pdb; pdb.set_trace()
    # weighted_loss_func = UncertaintyLoss(2, epoch=0).cuda()
    weighted_loss_func = AutomaticWeightedLoss(2).cuda()
    params = [
        {'params': fc_params, 'lr': 0.002},
        {'params': other_params, 'lr': 0.002},
        {'params': filter(lambda x: x.requires_grad, weighted_loss_func.parameters()), 'lr': 0.01}
    ]

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = getattr(criterions, args.criterion)
    # loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.7, 1])).float()).cuda()
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.7, 1])).float()).cuda()
    attention_loss = nn.MSELoss()
    target = torch.rand((1, 128, 128, 128))
    weight = torch.zeros_like(target)
    weight = torch.fill_(weight, 0.3)
    weight[target > 0] = 0.7
    seg_loss = nn.BCELoss(weight=weight.float().cuda(), size_average=True)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint',
                                      args.experiment + args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    resume = ''

    # writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    train_set = ADNI(train_list, train_root, args.mode)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    logging.info('Samples for train = {}'.format(len(train_set)))

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)

    valid_set = ADNI(valid_list, valid_root, 'test')
    logging.info('Sample for test = {}'.format(len(valid_set)))

    num_gpu = (len(args.gpu) + 1) // 2

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    best_acc = 0
    best_dice = 0
    training_info = []
    loss_list = []
    for epoch in range(args.start_epoch, args.end_epoch):
        model.train()
        train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()
        train_dice = []
        train_correct = 0
        train_num = 0
        y_predict = []
        y_true = []
        # i=1
        for i, data in enumerate(train_loader):
            # if i>1:
            #     continue
            # i = i+1
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target, cls = data  # [2,4,128,128,128] [2, 128, 128, 128] # 2->batch_size 4->modility
            cls = cls.squeeze(1)
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            cls = cls.cuda(args.local_rank, non_blocking=True)

            output, name, attention_map = model(x)  # [b, 1, 128, 128, 128], [b, 2] , [b, 2, 16, 16, 16]
            # get the most likely part of output and use it to activate the seg
            # 0 -> 0-class activate map
            # 1 -> 1-class activate map
            idx = torch.argmax(name, dim=1)
            attention_map = attention_map[:, idx, :, :, :]
            # 第二种思路，将原有的traget的维度降维到目标的大小
            compared_mask = F.interpolate(output, size=(16, 16, 16))
            # seg_loss
            seg_map = output.squeeze(1).float()
            # import pdb; pdb.set_trace()
            dice_loss = criterion(output, target)
            bceloss = seg_loss(seg_map, target.float())
            loss_seg = dice_loss + 0.5 * bceloss

            # cls_loss
            ce_loss = loss_function(name, cls)
            atten_loss = attention_loss(attention_map, compared_mask)  # 使用降维后的target和激活图进行比较，查看是否正确

            if (epoch + 1) > int(args.cls_start):

                loss_certainty = loss_seg + 1.5 * ce_loss + 0.5 * atten_loss
                loss_uncertainty = weighted_loss_func(loss_seg, ce_loss) + 0.5 * atten_loss
                loss_sum1 = all_reduce_tensor(loss_uncertainty, world_size=num_gpu).data.cpu().numpy()
                loss_sum2 = all_reduce_tensor(loss_certainty, world_size=num_gpu).data.cpu().numpy()
                dice_loss1 = all_reduce_tensor(loss_seg, world_size=num_gpu).data.cpu().numpy()
                ce_loss1 = all_reduce_tensor(ce_loss, world_size=num_gpu).data.cpu().numpy()
                atten_loss1 = all_reduce_tensor(atten_loss, world_size=num_gpu).data.cpu().numpy()
                loss = loss_uncertainty
                # loss_sum = dice_loss + 1.5 * ce_loss
            else:
                # loss_sum = dice_loss
                loss_certainty = loss_seg + 1.5 * ce_loss + 0.5 * atten_loss
                loss_uncertainty = weighted_loss_func(loss_seg, ce_loss) + 0.5 * atten_loss
                loss_sum1 = all_reduce_tensor(loss_uncertainty, world_size=num_gpu).data.cpu().numpy()
                loss_sum2 = all_reduce_tensor(loss_certainty, world_size=num_gpu).data.cpu().numpy()
                dice_loss1 = all_reduce_tensor(loss_seg, world_size=num_gpu).data.cpu().numpy()
                ce_loss1 = all_reduce_tensor(ce_loss, world_size=num_gpu).data.cpu().numpy()
                atten_loss1 = all_reduce_tensor(atten_loss, world_size=num_gpu).data.cpu().numpy()
                loss = loss_certainty

            # import pdb; pdb.set_trace()

            _, cls_result = torch.max(name.data, 1)
            # import pdb;pdb.set_trace()
            train_correct += (cls_result == cls.data).sum().item()
            train_num += len(cls_result)
            train_dice.append(meandice(output, target).cpu().detach().numpy())
            y_predict.extend(cls_result.data.cpu().tolist())
            y_true.extend(cls.data.cpu().tolist())
            # import pdb;
            # pdb.set_trace()
            if args.local_rank == 0:
                if i % 20 == 0:
                    logging.info(
                        'Epoch {} Iter:{} || uncertainty；{:.5f} certainty:{:.4f} || seg_loss: {:.5f}  cls_loss:{:.5f} ta_loss:{:.5f}||'
                        .format(epoch, i, loss_sum1, loss_sum2, dice_loss1, ce_loss1, atten_loss1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            if args.local_rank == 0:

                if (epoch + 1) % int(args.test_freq) == 0:
                    logging.info('-------------------------------test set------------------------------')
                    with torch.no_grad():
                        model.eval()
                        cls_corr = 0
                        cls_num = 0
                        train_acc = []
                        dice = []
                        IOU = []
                        test_num = 50
                        true_list, pred_list = [], []

                        dice_list, JC, VE, RECALL, PPV, JC, PRESION, HD95, ASD, RAVD = [], [], [], [], [], [], [], [], [], []
                        for i, data in enumerate(valid_loader):
                            msg = 'subject {}/{}, '.format(i + 1, len(valid_loader))
                            image, hippo_mask, cls_token = data
                            # load the data into cuda
                            image = image.cuda(non_blocking=True).unsqueeze(1)
                            hippo_mask = hippo_mask.cuda(non_blocking=True)
                            cls_token = cls_token.cuda(non_blocking=True)

                            mask, cls, attention_map_result = model(image)

                            mask[mask > 0.5] = 1
                            mask[mask < 0.5] = 0
                            dice.append(meandice(mask, hippo_mask).cpu().detach().numpy())
                            _, cls_result = torch.max(cls.data, 1)
                            # import pdb;pdb.set_trace()
                            cls_corr += (cls_result == cls_token).sum().item()
                            cls_num += 1
                            logging.info('{} real_time acc : {:.3f}%, average_dice:{:.5f}, this subject is:{}'
                                         .format(msg, cls_corr / cls_num * 100, np.mean(dice),
                                                 (cls_result == cls_token).item()))
                            pred_list.append(cls_result.data.cpu().tolist())
                            true_list.append(cls_token.data.squeeze().tolist())

                    f1 = f1_score(true_list, pred_list)
                    auc = roc_auc_score(true_list, pred_list)
                    recall = recall_score(true_list, pred_list)
                    precision = precision_score(true_list, pred_list)

                    # logging.info('Test set seg: average_dice:{:.4f}  ppv:{:.4f} JC:{:.4f} RAVD:{:.4f}, HD:{:.4f}'
                    #              .format(np.mean(dice_list), np.mean(PPV), np.mean(JC), np.mean(RAVD), np.mean(HD95)))
                    logging.info('training acc is {:.3f} f1:{:.4f} auc:{:.4f} recall(sen):{:.4f} '
                                 'precision(spe):{:.4f} '.format(cls_corr / cls_num * 100, f1, auc, recall, precision))
                    if cls_corr / cls_num * 100 > best_acc:
                        best_acc = cls_corr / cls_num * 100
                        file_name = os.path.join(checkpoint_dir, 'best_acc_checkpoint.pth'.format(epoch))
                        torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict(),
                        },
                            file_name)
                    if np.mean(dice) > best_dice:
                        best_dice = np.mean(dice)
                        file_name = os.path.join(checkpoint_dir, 'best_dice_checkpoint.pth'.format(epoch))
                        torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict(),
                        },
                            file_name)

                    logging.info('------------------------------test end------------------------------')
        if args.local_rank == 0:
            f1 = f1_score(y_true, y_predict)
            auc = roc_auc_score(y_true, y_predict)
            recall = recall_score(y_true, y_predict)
            precision = precision_score(y_true, y_predict)
            training_info.append(
                [epoch, cls_corr / cls_num * 100, recall, precision, auc, ])

            logging.info('EPOCH: {}--> training_set acc is {:.3f} f1:{:.2f} auc:{:.2f} recall:{:.2f} '
                         'precision:{:.2f} '.format(epoch, 100 * train_correct / train_num, f1, auc, recall, precision))
            logging.info('EPOCH: {}--> test_set_dice:{:.3f}, best_acc:{:.2f}, best_dice:{:.2f}'
                         .format(epoch, np.mean(train_dice), best_acc, best_dice))
            epoch_time_minute = (end_epoch - start_epoch) / 60
            remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        # writer.close()
        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))
    import pandas as pd
    training_info = pd.DataFrame(training_info,
                                 columns=['Epoch', 'Acc', 'F1', 'auc', 'recall', 'precision', 'epoch_dice'])
    training_info.to_csv('./training_info.csv', index=False)
    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def meandice(pred, label):
    sumdice = 0
    smooth = 1e-6

    pred_bin = pred
    label_bin = label

    pred_bin = pred_bin.contiguous().view(pred_bin.shape[0], -1)
    label_bin = label_bin.contiguous().view(label_bin.shape[0], -1)

    intersection = (pred_bin * label_bin).sum()
    dice = (2. * intersection + smooth) / (pred_bin.sum() + label_bin.sum() + smooth)
    sumdice += dice

    return sumdice


class UncertaintyLoss(nn.Module):

    def __init__(self, v_num, epoch):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.tensor([1.0, 1.5, 0.5])
        # import pdb; pdb.set_trace()
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num
        self.epoch = epoch

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        if self.epoch % 200 == 0:
            logging.info(
                'weight of three tasks:seg:{:.4f}, cls:{:.4f}, attn:{:.4f}'.format(self.sigma[0], self.sigma[1],
                                                                                   self.sigma[2]))
        self.epoch += 1
        return loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        # params = torch.tensor([1, 1.5, 0.1], requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.epoch = 0

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        if self.epoch % 200 == 0:
            logging.info(
                # 'weight of tasks:seg:{:.4f}, cls:{:.4f} attn_loss:{:.4f}'.format(self.params[0], self.params[1], self.params[2]))
                'weight of tasks:seg:{:.4f}, cls:{:.4f} '.format(self.params[0], self.params[1],
                                                                 ))
        self.epoch += 1
        return loss_sum


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    ppv = metric.binary.positive_predictive_value(pred, gt)
    ravd = metric.binary.ravd(pred, gt)
    return dice, ppv, jc, ravd, hd


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
