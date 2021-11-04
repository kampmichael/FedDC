import torch
import torchvision
from torchvision import transforms, datasets
import torch.utils
import numpy as np
import random
import itertools
from collections import Counter

traindir = r'data/preprocessed/train/'
valdir = r'data/preprocessed/val/'


def getPneum(device):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    trainloader = datasets.ImageFolder(traindir, transform=train_transforms)
    testloader = datasets.ImageFolder(valdir, transform=val_transforms)
    #X_train = np.array([sample[0].numpy() for sample in itertools.islice(trainloader,8)])
    #y_train = np.array([sample[1] for sample in itertools.islice(trainloader,8)]).astype(np.float64).reshape(-1,1)
    #X_test = np.array([sample[0].numpy() for sample in itertools.islice(testloader,8)])
    #y_test = np.array([sample[1] for sample in itertools.islice(testloader,8)]).astype(np.float64).reshape(-1,1)
    X_train = np.array([sample[0].numpy() for sample in trainloader])
    y_train = np.array([sample[1] for sample in trainloader]).astype(np.int_)
    #y_train = np.array([sample[1] for sample in trainloader]).astype(np.float64).reshape(-1,1)


    X_test = np.array([sample[0].numpy() for sample in testloader])
    y_test = np.array([sample[1] for sample in testloader]).astype(np.int_)
    #y_test = np.array([sample[1] for sample in testloader]).astype(np.float64).reshape(-1,1)

    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    X_train = torch.cuda.FloatTensor(X_train)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.cuda.FloatTensor(X_test)
    y_test = torch.tensor(y_test, device=device)
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

def splitIntoLocalDataLimClasses(X, y, m, n_local, rng, numClasses):

    ## generate different stacks of samples, one per class
    idcs = [ np.array([yIdx for yIdx, yVal in enumerate(y) if yVal == i]) for i in range(10) ]
    for i in range(10):
        rng.shuffle(idcs[i])

    ## offsets in these stacks
    idcsOffsets = np.repeat(0,10)

    Xs, Ys = [],[]
    for i in range(m):

        ## determine which classes to cover
        class_pick = np.arange(10)
        rng.shuffle(class_pick)
        class_pick = class_pick[:numClasses]
        ## generate assignment of samples to classes (uniform at random)
        classIdcs = [ class_pick[c] for c in np.random.random_integers(0, numClasses-1, n_local) ]
        ## get the sample indices
        sample = np.concatenate([idcs[cl][idcsOffsets[cl]:(idcsOffsets[cl]+count)] for cl, count in Counter(classIdcs).items()])

        ## append Xs and Ys
        Xs.append(X[sample,:,:,:])
        Ys.append(y[sample])

        for cl, count in Counter(classIdcs).items():
            idcsOffsets[cl] += count

    return Xs, Ys


def getPneumDataLoader(batch_size, num_clients, n_local):

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply(transforms=[transforms.RandomAffine(
                               degrees=(-180, 180), translate=(0.10, 0.10), scale=(0.9, 1.10))], p=0.99),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    trainset = datasets.ImageFolder(traindir, transform=train_transforms)
    n = len(trainset)
    trainset = torch.utils.data.Subset(trainset,
        random.sample(range(n), min(num_clients*n_local, n))
    )
    
    testset = datasets.ImageFolder(valdir, transform=val_transforms)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        pin_memory=False, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        pin_memory=False, drop_last=False)
 

    classes = ('healthy', 'pneumonia')

    return trainloader, testloader, classes
