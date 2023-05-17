import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torch.optim
import argparse
import os
import random
import numpy as np
import time
import setproctitle
import torch
import torch.optim
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from models.TransBTS.FCN import TransBTS
# from models.TransBTS.trans384dim import TransBTS
from torch.utils.data import DataLoader
from data.ADNI import ADNI
from M3d_Cam.cam import medcam
import nibabel as nib
import torch.nn.functional as F

from medpy import metric
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, recall_score, precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--user', default='leizhenxin', type=str)
parser.add_argument('--root', default='PATH/TO/DATASET', type=str)
parser.add_argument('--valid_dir', default='', type=str)
parser.add_argument('--valid_file', default='test_data.txt', type=str)
parser.add_argument('--output_dir', default='output', type=str)
parser.add_argument('--submission', default='submission', type=str)
parser.add_argument('--visual', default='visualization', type=str)
parser.add_argument('--experiment', default='', type=str)
parser.add_argument('--test_date', default='', type=str)
parser.add_argument('--test_file', default='', type=str)
parser.add_argument('--use_TTA', default=True, type=bool)
parser.add_argument('--post_process', default=True, type=bool)
parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)
parser.add_argument('--crop_H', default=128, type=int)
parser.add_argument('--crop_W', default=128, type=int)
parser.add_argument('--crop_D', default=128, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--model_name', default='TransBTS', type=str)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--outputpath', default='/hy-tmp/test_result', type=str)

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    # print(medcam.get_layers(model))
    # model = medcam.inject(model, backend='gcam', output_dir='attention_map', save_maps=True, layer='GAP',
    #                       return_attention=True)
    model = torch.nn.DataParallel(model).cuda()

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', 'CHECKPOINT', 'best_acc_checkpoint.pth')
    
    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(os.path.join(args.experiment + args.test_date, args.test_file)))
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = ADNI(valid_list, valid_root, mode='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment + args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment + args.test_date)
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()
    with torch.no_grad():
        model.eval()
        cls_corr = 0
        cls_num = 0
        train_acc = []
        # dice_list = []
        predict_list = []
        true_list = []
        dice_list, JC, VE, RECALL, PPV, JC, PRESION, HD95, ASD, RAVD = [], [], [], [], [], [], [], [], [], []
        IOU = []
        test_num = 50
        for i, data in enumerate(valid_loader):
            msg = 'subject {}/{}, '.format(i + 1, len(valid_loader))
            image, hippo_mask, cls_token = data
            # load the data into cuda
            image = image.cuda(non_blocking=True).unsqueeze(1)
            hippo_mask = hippo_mask.cuda(non_blocking=True)
            cls_token = cls_token.cuda(non_blocking=True)

            mask, cls, attention_map = model(image)
            # cls, mask = model(image)
            # cls, attention_map = cls

            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            idx = torch.argmax(cls, dim=1)
            attention_map = attention_map[:, idx, :, :, :]
            # dice = meandice(mask, hippo_mask).cpu().detach().numpy()

            compared_mask = F.interpolate(mask, size=(16, 16, 16))

            # 分割指标计算
            mask = mask.view(128, 128, 128).cpu().detach().numpy()
            hippo_mask = hippo_mask.view(128, 128, 128).cpu().detach().numpy()
            dice, ppv, jc, ravd, hd = calculate_metric_percase(mask, hippo_mask)

            dice_list.append(dice)
            JC.append(jc)
            PPV.append(ppv)
            HD95.append(hd)
            RAVD.append(ravd)
            # 分类指标计算
            _, cls_result = torch.max(cls.data, 1)

            if cls_result == 1:
                mark = 'CN'
            elif cls_result == 0:
                mark = 'AD'

            if cls_token.data == 1:
                tar_mark = 'CN'
            elif cls_token.data == 0:
                tar_mark = 'AD'
            # import pdb;pdb.set_trace()
            cls_corr += (cls_result == cls_token).sum().item()
            cls_num += 1
            predict_list.append(cls_result.data.cpu().tolist())
            true_list.append(cls_token.data.squeeze().tolist())

            # 保存分割分类的信息
            # print('Saving the result to'+os.path.join(args.outputpath, ('subject_' + str(i)))
            # if not os.path.exists(os.path.join(args.outputpath, ('subject_' + str(i)))):
            #     os.makedirs(os.path.join(args.outputpath, ('subject_' + str(i))))
            # image_view = image.view(128, 128, 128).cpu().detach().numpy()
            # hippo_mask_view = hippo_mask.view(128, 128, 128).cpu().detach().numpy().astype(float)
            # mask_view = mask.view(128, 128, 128).cpu().detach().numpy()
            # cmp_mask = compared_mask.view(16, 16, 16).cpu().detach().numpy()
            # attention_map_view = attention_map.view(16, 16, 16).cpu().detach().numpy()
            #

            # # intensity_image = image_view + attention_map_view
            # # import pdb;
            # # pdb.set_trace()
            # origin = nib.Nifti1Image(image_view, np.eye(4))
            # nib.save(origin, os.path.join(args.outputpath, ('subject_' + str(i)), 'image.nii'))
            # # hippo_mask_view[hippo_mask_view == 1] = 2
            # hippo_mask_view = nib.Nifti1Image(hippo_mask_view, np.eye(4))
            # nib.save(hippo_mask_view, os.path.join(args.outputpath, ('subject_' + str(i)), 'target_mask.nii'))
            #
            # cmp_mask = nib.Nifti1Image(cmp_mask, np.eye(4))
            # nib.save(cmp_mask, os.path.join(args.outputpath, ('subject_' + str(i)), 'cmp_mask.nii'))
            #
            # hp_mask = nib.Nifti1Image(mask_view, np.eye(4))
            # nib.save(hp_mask, os.path.join(args.outputpath, ('subject_' + str(i)), 'mask.nii'))
            #
            # attention_map = nib.Nifti1Image(attention_map_view, np.eye(4))
            # nib.save(attention_map, os.path.join(args.outputpath, ('subject_' + str(i)), 'attention_map.nii'))
            # intensity_image = nib.Nifti1Image(intensity_image, np.eye(4))
            # nib.save(intensity_image, os.path.join(args.outputpath, ('subject_' + str(i)), 'intensity_map.nii'))

            # import pdb; pdb.set_trace()

            print('{}, dice:{:.4f}, acc:{:.2f}%, this subject is {}, target is {}, {}'
                  .format(msg, dice, cls_corr / cls_num * 100, mark, tar_mark, (cls_result == cls_token).item()))

    # 提取计算指标
    f1 = f1_score(true_list, predict_list)
    # roc = roc_curve(y_true, y_predict)
    auc = roc_auc_score(true_list, predict_list)
    recall = recall_score(true_list, predict_list)
    precision = precision_score(true_list, predict_list)
    # import pdb; pdb.set_trace()
    print('Test set seg: average_dice:{:.4f}  ppv:{:.4f} JC:{:.4f} RAVD:{:.4f}, HD:{:.4f}'
          .format(np.mean(dice_list), np.mean(PPV), np.mean(JC), np.mean(RAVD), np.mean(HD95)))
    print('training acc is {:.3f} f1:{:.4f} auc:{:.4f} recall(SEN):{:.4f} '
          'precision(SPE):{:.4f} '.format(cls_corr / cls_num * 100, f1, auc, recall, precision))

    print('Test dataset accuracy is:{:.4f}%, average_dice is:{:.4f}'
          .format(cls_corr / cls_num * 100, np.mean(dice_list)))
    end_time = time.time()
    full_test_time = (end_time - start_time) / 60
    average_time = full_test_time / len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


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


def recall(predict, target):  # Sensitivity, Recall, true positive rate都一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # precision = metric.binary.precision(pred, gt)
    # recall = metric.binary.recall(pred, gt)
    ppv = metric.binary.positive_predictive_value(pred, gt)
    ravd = metric.binary.ravd(pred, gt)
    # vc = metric.binary.volume_correlation(pred, gt)
    return dice, ppv, jc, ravd, hd


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()
