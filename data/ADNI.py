import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import nibabel

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)
        return {'image': image, 'label': label, 'cls': sample['cls']}


class resample(object):
    """resample the file from 256*256*256 to 200*200*200 to better cover the label"""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = image[28:228, 28:228, 0:200, ...]
        label = label[..., 28:228, 28:228, 0:200]
        return {'image': image, 'label': label, 'cls': sample['cls']}

class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label, 'cls':sample['cls']}


class center_crop(object):
    """this function is used in valid model to generate resized image"""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = image[64:192, 64:192, 64:192, ...]
        label = label[..., 64:192, 64:192, 64:192]
        return {'image': image, 'label': label, 'cls': sample['cls']}


class Random_Flip(object):
    
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)
        return {'image': image, 'label': label, 'cls': sample['cls']}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 200 - 128)
        W = random.randint(0, 200 - 128)
        D = random.randint(0, 200 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]
        return {'image': image, 'label': label, 'cls': sample['cls']}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0 - factor, 1.0 + factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image * scale_factor + shift_factor
        return {'image': image, 'label': label, 'cls': sample['cls']}


class label_norm(object):
    '''
        the label is processed with freesurfer, its intensity equals to 128
        to process the label, we need to convert it to 1
    '''
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        label[label == 128] = 1
        return {'image': image, 'label': label, 'cls': sample['cls']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        
        image = np.ascontiguousarray(image)
        label = sample['label']
        label = np.ascontiguousarray(label)
        cls_label = sample['cls']
        cls_label = np.ascontiguousarray(cls_label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        cls_label = torch.from_numpy(cls_label).long()

        return {'image': image, 'label': label, 'cls': cls_label}


def transform(sample):
    trans = transforms.Compose([
        # Pad(),
        # Random_rotate(),  # time-consuming
        # MaxMinNormalization(),
        resample(),
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        label_norm(),
        ToTensor()
    ])

    return trans(sample)


def transform_test(sample):
    trans = transforms.Compose([
        # Pad(),
        # MaxMinNormalization(),
        center_crop(),
        # Random_Crop(),
        label_norm(),
        ToTensor()
    ])

    return trans(sample)


class ADNI(Dataset):
    def __init__(self, list_file, root='', mode='train', testset='AD'):
        self.lines = []
        paths, names, cls = [], [], []
        self.AD, self.CN, self.MCI = [], [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split(',')[0]
                cls_label = line.split(',')[1]
                if cls_label == testset:
                    cls_label = 0
                    self.MCI.append(1)
                elif cls_label == 'CN':
                    cls_label = 1
                    self.CN.append(1)
                else:
                    continue
                names.append(name)
                cls.append([name, cls_label])
                path = os.path.join(root, name)
                paths.append(path)
                self.lines.append(line)

        cls = np.array(cls)
        self.mode = mode
        self.names = names
        self.paths = paths
        self.cls = cls


    def __getitem__(self, item):
        path = self.paths[item]
        
        if self.mode == 'train':
            image, label = pkload(path + '/data_f32b0.pkl')
            np.nan_to_num(image, copy=False, nan=0.0)
            cls_label = int(self.cls[item][1])

            sample = {'image': image, 'label': label, 'cls': cls_label}
            sample = transform(sample)
            return sample['image'], sample['label'], sample['cls']
        elif self.mode == 'test':
            image, label = pkload(path + '/data_f32b0.pkl')
            cls_label = int(self.cls[item][1])

            sample = {'image': image, 'label': label, 'cls': cls_label}
            sample = transform_test(sample)
            return sample['image'], sample['label'], sample['cls']

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
