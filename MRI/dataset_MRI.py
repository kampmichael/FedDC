import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import os

from collections import Counter

class MRI(Dataset):
    def __init__(self, path):
        self.paths = []
        for dirname, _, filenames in os.walk(path):
            for filename in sorted(filenames):
               self.paths.append(os.path.join(dirname, filename))
        self.size=len(self.paths)
        
        
    def __len__(self):
        return self.size
    
    def get(self, ix, device, size, pretrained):
        im = cv2.imread(self.paths[ix])[:,:,::-1]
        im = cv2.resize(im,size)
        im = im / 255.
        im = torch.cuda.FloatTensor(im, device=device)
        im = im.permute(2,0,1)
        #if pretrained:
        #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #    im = normalize(im)
        label = self.paths[ix].split("/")[-2]
        if label == "no":
            target = 0
        if label == "yes":
            target = 1
        return im, label, target
    
    def __getitem__(self, ix, device, size=(150,150), pretrained=False):
        im, label, target = self.get(ix, device, size, pretrained)
        return im.to(device).float(),torch.tensor(int(target)).to(device).long()
        
    def getDataset(self, device, size=(150,150), pretrained=False):
        imgs = []
        labels = []
        targets = []
        for ix in range(self.size):
            im, label, target = self.get(ix, device, size, pretrained)
            imgs.append(im)
            labels.append(label)
            targets.append(target)
        X = torch.stack(imgs)
        y = torch.tensor(np.array(targets), device=device)
        return X, y 
            
        

def getMRI(device, path, rng, size=(150,150), pretrained=False):

    dataset = MRI(path)
    X, y = dataset.getDataset(device, size, pretrained)

    ## shuffle
    idxs = np.arange(y.shape[0])
    rng.shuffle(idxs)
    X = X[idxs,:,:,:]
    y = y[idxs]
    ## split
    X_train = X[:200,:,:,:]
    y_train = y[:200]
    X_test = X[200:,:,:,:]
    y_test = y[200:]

    return X_train, y_train, X_test, y_test


def splitIntoLocalData(n, m, n_local, rng):
    if m*n_local > n:
        print("Error: not enough data (n=",n,") for",m,"sets of size ",n_local,". Reducing local size to",n//m,".")
        n_local = n // m
        
    idxs = np.arange(n)
    rng.shuffle(idxs)
    bucket_size = n_local
    i = 0
    client_idxs = []
    while (i+1)*bucket_size <= n and i < m:
        idx = idxs[i*bucket_size:(i+1)*bucket_size]
        client_idxs.append(idx)
        i += 1
    return client_idxs


def getSample(client_idxs, batchSize, rng):
    n = len(client_idxs)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    sample_idxs = np.array([client_idxs[idxs[j]] for j in range(batchSize)])
    return sample_idxs



def getMRIDataLoader(device, path, rng, batch_size, num_clients, n_local, size=(150,150), pretrained=False):

    dataset = MRI(path)
    X, y = dataset.getDataset(device, size, pretrained)

    ## shuffle
    idxs = np.arange(y.shape[0])
    rng.shuffle(idxs)
    X = X[idxs,:,:,:]
    y = y[idxs]
    ## split
    X_train = X[:200,:,:,:]
    y_train = y[:200]
    X_test = X[200:,:,:,:]
    y_test = y[200:]

    
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    trainloader = DataLoader(train_data, batch_size = batch_size,shuffle = True)
    testloader = DataLoader(test_data, batch_size = batch_size,shuffle = False)
    
    classes = ('no','yes') #brain tumor detected, no or yes
    return trainloader, testloader, classes
