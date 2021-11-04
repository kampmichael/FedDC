import numpy as np
import torch


def trainEvalLoopVanilla(model, loss, optimizer, scheduler, trainloader, testloader, epochs, device, log_interval=10):

    for epoch in range(epochs):

        trainVanillaModel(model, loss, optimizer, trainloader, epoch, device, log_interval)
        evaluateVanillaModel(model, loss, testloader, device)
        if scheduler is not None:
            scheduler.step()



def trainVanillaModel(model, loss, optimizer, dataloader, epoch, device, log_interval):

    model.train()
    totalLoss = 0
    correct1 = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, pred = output.topk(1, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct1 += correct[:,:1].sum().item()
        train_loss = loss(output, target)
        totalLoss += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop1 Accuracy: {:.1f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item(),
                100. * correct1 /((batch_idx+1) * len(data))))
    return totalLoss / len(dataloader.dataset), 100. * correct1 / len(dataloader.dataset)




def evaluateVanillaModel(model, loss, dataloader, device):

    model.eval()
    totalLoss = 0
    correct1 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            totalLoss += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
    average_loss = totalLoss / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1