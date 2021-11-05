# COVID-19 detector from Voice

###
This package is developed for detecting COVID-19 from voice samples. 
A total of 6 voice samples are collected from each speaker. 
These voice samples are - vowel-i, vowel-u, vowel-a, alphabet, count 1-20 and cough 

### Here is how this package works : - 

1. Collect input audio files and organize them according to speaker id. 
2. Extract Pase features on the input audio file. 
3. Run CNN Classification on the pase extracted feature file for the audio data. 

### Steps to setup the package

It is recommended to create a new environment in conda instead of running in your base path of conda. 
Run all the below commands in a sequential order -- Without the sequential order environment setup may break/lead to incompatible packages

```
conda create --name covid_cnn_v1 python=3.6
conda activate covid_cnn_v1
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -c conda-forge librosa
pip install progressbar
pip install cupy-cuda110
pip install git+https://github.com/detly/gammatone
pip install cupy-cuda101 pynvrtc git+https://github.com/salesforce/pytorch-qrnn
pip install pynvrtc==8.0
git clone https://github.com/santi-pdp/pase -- follow steps to setup --> python setup.py install
conda install pandas
```

#### Other steps for the feature extractor
```
Download model for feature extractor here --> https://drive.google.com/file/d/1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW/view 
Place it in the root directory of the github repository 
```

#### Sample data format 
```
Sample positive speaker id csv 
UPID
UPID-65428b55
UPID-65428b56

Similarly negative speaker id is generated as well.
```

```
Data is placed in version_1/data folder
Data is partitioned into positive and negative folders -- where positive refers to infected with COVID-19 patient

Sample file name is - cough_UPID-9ae17732-20200611172756.wav
Format of filename should be --> {Type_of_audio}_{speaker_id}_{date_collected}.wav 

Type of audio can be vowel-i, vowel-u, vowel-a, alphabet, count 1-20 and cough
Speaker id - has to be a unique identifier per speaker
Date collected -- any format in which date is mentioned is fine - should not contain '_' in the date part and contain just continuous string of numbers. 
```

## Instructions for running the package
This assumes that your environment was correctly setup and you are running from the conda environment. 
1. python pase_feature_extractor.py data/positive/positive.list data/negative/negative.list features/  --- This command generates features from pase model
2. Create a list of the features generated and save it as a ctl file -- Better to have fullfile path in this list.
3. bash run_expts.sh < path to the above saved ctl file >

### Comments when running
```
By default the code of paseCNN.py assumes you want to run both training and testing. 
In order to run only inference --> the training mode argument needs to be set as 0 [it is set as 1 in code for training+testing] and appropriate model needs to be loaded
Other hyperparameter arguments in the code for running can be passed via the run_expts.sh script 
```

#### processing.py
loads the data, model and processes everything to print out the speaker and utterance wise AUC ROC. 

#### fobi.py
FOBI ICA (Independent component analysis) algorithm - Not being used in the current analysis

#### gen_plots.py
Generates and stores the ROC/Precision-Recall graphs. 

conda install pytorch==1.4.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch -- Code can work with this package too -- but needs modification

### Comments and pointers
This pipeline is able to generate AUC (Area under the ROC curve) of over 0.8 without data augmentation and data augmentation can be done in many ways for audio files -- A good repository to review for the augmentation approaches is https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6 and https://github.com/iver56/audiomentations
Augmentation leads to about 2.5-3% improvement in the AUC scores. 
