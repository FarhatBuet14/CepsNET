## CepsNET: Abnormal Heart Sound Detection Considering Additive and Convolutional Distortion using Fused Cepstral Features

We propose to use a fusion of Mel-FrequencyCepstral Coefficients (MFCC) and its variants as the input to a 2D-Residual Neural Network architecture to address the presence of channel and additive noise distortion simultaneously.

## Parametric Model of the Stethoscope

![signal.png](https://github.com/FarhatBuet14/Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features/blob/main/images/signal.png)

## Multiple Cepstral Features
* fbank
* log-fbank
* mfcc_26
* mfcc_13
* fbank_log-fbank
* fbank_mfcc_13
* log-fbank_mfcc_13
* fbank_log-fbank_mfcc_13
* mfcc_13_d
* mfcc_13_dd

![MFCC_flow.png](https://github.com/FarhatBuet14/Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features/blob/main/images/MFCC_flow.png)

## Proposed Model Architechture

![model%20architechture.png](https://github.com/FarhatBuet14/Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features/blob/main/images/model%20architechture.png)

## Experimental Results

* Validation Scores of Multiple Trainings 

![Result_Table.png](https://github.com/FarhatBuet14/Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features/blob/main/images/Result_Table.png)

* Consequence of Domain Balanced Training (DBT)

![tsne.png](https://github.com/FarhatBuet14/Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features/blob/main/images/tsne.png)

## Requirements
* Python 3.8.5
* Matlab 2017b

## Installation
~~~~{.python}
git clone https://github.com/FarhatBuet14/Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features.git
cd Abnormal-Heart-Sound-Detection-using-Fused-Cepstral-Features/codes
pip install -r requirements.txt
~~~~


## How To Run

#### Data Preparation:

1. First download the *data* folder from this [GoogleDrive Link](https://drive.google.com/open?id=1MPBhemO6XeDfjIm5-SOQUGvmzIl0Hx03)<br />
Place [Physionet dataset](https://physionet.org/content/challenge-2016/1.0.0/#files) (not included in the provided *data* folder) in the corresponding folders inside the *data/physionet/training* folder.
The csv files containing the labels should be put inside the corresponding folders inside the *labels* folder and all of them should have the same name, currently 'REFERENCE_withSQI.csv'. If you change the name you'll have to rename the variable *labelpath* in  *extract_segments.m*<br /> 
2. Run *extract_segments.m* it first then run *data_fold.m* to create data fold in *mat* format which will be loaded by the model for training and testing. *fold_0.mat* is given inside *data/feature/folds* for convenience, so that you don't have to download the whole physionet dataset and extract data for training and testing.

3. For preparing features, run *prepare_cepstralFeature.py* with passing the name of the input feature (Select from the given list) as an argument. It will prepare an .npz file, saved in the "data/feature/train_val_npzs" folder. For example,
~~~~{.python}
python prepare_cepstralFeature.py --inp log-fbank
~~~~

#### Training:
For Training run the *trainer.py* and provide a dataset name (or fold name) i.e. *fold_0*. The command should be like this : 
~~~~{.python}
python train.py fold_0
~~~~
Other parameters can be passed as arguments. 
~~~~{.python}
python train.py fold_0 --ep 300 --batch 1000 
~~~~


#### Re-Generate Results or Validate with Pretrained Models:
There is given a validation.ipynb file to validate the pretrained models with different types of input features. "pretrained.xlxs" is given in the "log" folder which preserves the hypyer-parameter of all the pretrained models. 

Change the "feature" variable in the "Hyper-parameters" block and obeserve the validation result.

#### McNemar’s Test
We have also provided "testbench.ipynb" file which contains the codes for comparing the results with McNemar’s Test. Put any two features in the "Selected Feature List" and run the "McNemar’s Test" block to compare.
