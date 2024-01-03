# FedDC

This repository contains the implementation of federated daisy-chaining and the code for reproducing the experiments in the paper
Anonymous. "Picking Daisies in Private: Federated Learning from Small Datasets"

The following experiments are available:
* CIFAR10
* MNIST
* MRI
* Pneumonia
* Radon Machine on SUSY
* Synthetic Data

## CIFAR10
To execute FedDC on CIFAR10, training resnet18 via model averaging, execute feddc_CIFAR10_pytorch.py. 
The experiments can be reproduced by executing the script runExp.sh.

## MRI
To execute FedDC on the MRI dataset, training a custom MLP via model averaging, execute feddc_MRI_pytorch.py. 
The experiments can be reproduced by executing the script runExp.sh.

## Pneumonia
To execute FedDC on the pneumonia dataset, training resnet18 via model averaging, execute feddc_pneum_pytorch.py. 
The experiments can be reproduced by executing the script runExp.sh.

## Radon Machine on SUSY
To execute FedDC to train linear models on the SUSY dataset via the Radon Machine, execute feddc.py. 
To execute only the Radon machine, execute radonOnly.py.
The experiments can be reproduced by executing the scripts runFedDC_SUSY.script and runRadonOnly_SUSY.script.

## Synthetic Data
The experiments on synthetic data are summarized in the jupyter notebook SyntheticData.ipynb.

## Datasets
**CIFAR10**: We use the torchvision version of CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html) which downloads the dataset on demand.

**MNIST**: We use the torchvision version of MNIST [https://www.cs.toronto.edu/~kriz/cifar.html](https://yann.lecun.com/exdb/mnist/) which downloads the dataset on demand.

**MRI**: The dataset can be downloaded from kaggle: [kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). Extract the folder *brain_tumor_dataset*.
You can specify the path to the dataset using the `--dataset-path` argument.

**Pneumonia**: The dataset can be downloaded from kaggle: [kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). Please use the `PrepareData.ipynb` to copy images into a folder structure that separates train and validation data, as well as healthy and pneumonia images. You can specify the path to the dataset using the `--dataset-path` argument.

**SUSY**: The SUSY dataset can be downloaded from the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/279/susy](https://archive.ics.uci.edu/dataset/279/susy). Please extract the *SUSY.csv* file into the folder *data/*.
