# An Explainable Lesion-guided Multi-task Learning For Alzheimer;s Disease Diagnosis


## Flowchart of our DeepTAAD
<div align="center">
  <img width="100%" alt="DEEPTAAD illustration" src="fig/model.png">
</div>

## Classification result of our method
<div align="center">
  <img width="80%" alt="DEEPTAAD illustration" src="fig/result_table.png">
</div>
The roc curve of our model are shown as follow:
<div align="center">
  <img width="50%" alt="DEEPTAAD illustration" src="fig/rooc.png">
</div>

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel
- setproctitle
- medcam
- medpy

## Training
Run the training script on the ADNI dataset. Distributed training is available for training the proposed DeepTAAD, where --nproc_per_node decides the numer of gpus and --master_port implys the port number.
you can use the following command to train the model with task assistance loss
`CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 20003 train.py`

## Testing 
If  you want to test the model which has been trained on the ADNI dataset, run the testing script as following.
`python test_ADNI.py`


