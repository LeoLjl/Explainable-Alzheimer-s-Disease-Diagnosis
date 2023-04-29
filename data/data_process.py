"""
    this script is used to extract masked brain and hippocampus mask files from the dataset and convert it
    to .pkl inorder to better load from the ssd
"""

import pickle
import os
import numpy as np
import nibabel as nib

# modalities = ('flair', 't1ce', 't1', 't2')
file_name = ('brainmask', 'hippo_mask')

# train
train_set = {
    'root': r'G:\数据盘\dataset\HarP_origin',
    'flist': r'G:\数据盘\dataset\HarP_origin\train_HarP.txt',
    'has_label': True
}

# test/validation data
valid_set = {
    'root': '/Volumes/Untitled/dataset/split_dir/result_dir/ADNI_traindata',
    'flist': './valid.txt',
    'has_label': True
}

test_set = {
    'root': '/Volumes/Untitled/dataset/split_dir/result_dir/ADNI_traindata',
    'flist': './test_data.txt',
    'has_label': True
}


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'hippo_mask.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in file_name], -1)  # [240,240,155]

    output = path + 'data_.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


def process_f32b0(path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # print(path)
    subject_id = path.split('\\')[-1]
    print(subject_id)
    output_path = r'G:\数据盘\dataset\HarP_Processed'
    if not os.path.exists(os.path.join(output_path, subject_id)):
        os.makedirs(os.path.join(output_path, subject_id))
    else:
        print(os.path.join(output_path, subject_id) + ' has been processed!')
        return

    if has_label:
        label = np.array(nib_load(path + '/label.nii'), dtype='uint8', order='C')
    
    images = np.array(nib_load(path + '/brainmask.nii'), dtype='float32', order='C')
    output = os.path.join(output_path, subject_id) + '/data_f32b0.pkl'
    mask = images.sum(-1) > 0
    for k in range(4):
        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)
        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)
    if not has_label:
        return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = dset['flist']
    subjects = open(file_list).read().splitlines()
    names = [sub.split(',')[0] for sub in subjects]
    paths = [os.path.join(root, name) for name in names]
    
    for path in paths:
        process_f32b0(path, has_label)


if __name__ == '__main__':
    doit(train_set)
