# An Explainable Multi-task Transformer Network for Accurate Alzheimer's Disease Diagnosis
This repo contains the code to our paper: An Explainable Multi-task Transformer Network for Accurate Alzheimer's Disease Diagnosis.

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel
- setproctitle
- medcam

## Data Acquisition
- Dataset can be download from baidunetdisk, link will be posted soon

## Training
Run the training script on ADNI dataset. Distributed training is available for training the proposed TransAD, where --nproc_per_node decides the numer of gpus and --master_port implys the port number.
The cam_train.py file is used to generate the cam result during training

`python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 20003 train.py`
`CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 20003 cam_train.py`

## Testing 
If you want to test the model which has been trained on the ADNI dataset, change Line62 in `test_ADNI.py` and run the script as follows.

`python3 test_ADNI.py`


