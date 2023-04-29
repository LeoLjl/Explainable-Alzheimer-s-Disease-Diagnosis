## ADNI Date preprocess
In order to load the dile quicker, data_process.py is used to extect the label and original image from the dataset
you can use the command as follow

`python data_process.py`

After converting , the date can be load through by ADNI.py,which include data enhance and label extract.
you can use 

`python test_adni.py` 

to test the dataset class. The preprocessed dataset has already been done.

The test_data.txt and train_data.txt is the training list and valid list of dataset.
