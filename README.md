# FedDC

This repository contains the implementation of federated daisy-chaining and the code for reproducing the experiments in the paper
Anonymous. "Picking Daisies in Private: Federated Learning from Small Datasets"

The following experiments are available:
* CIFAR10
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