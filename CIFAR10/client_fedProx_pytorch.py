from parameters_pytorch import PyTorchNNParameters

import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class PyTorchNNFedProx():
    def __init__(self, batchSize, mu, mode, device, name = "PyTorchNN"):
        self._core          		= None
        self._flattenReferenceParams    = None
        self._mode			= mode
        self._device			= device
        self.name = name
        self.mu = mu
        self._avgModel = None

    def setCore(self, network):
        self._core = network
        self._avgModel = network.__class__() #works only for networks instantiated without arguments...

    def setModel(self, param: PyTorchNNParameters, setReference: bool):
        super(PyTorchNN, self).setModel(param, setReference)
        
        if setReference:
            self._flattenReferenceParams = self._flattenParameters(param)

    def setLoss(self, lossFunction):
        self._loss = eval("nn." + lossFunction + "()")

    def setUpdateRule(self, updateRule, learningRate, schedule_ep, schedule_changerate, weight_decay, **kwargs):
        additional_params = ""
        for k in kwargs:
            additional_params += ", " + k  + "=" + str(kwargs.get(k))
        self._updateRule = eval("optim." + updateRule + "(self._core.parameters(), lr=" + str(learningRate) + additional_params + ")")
        if (schedule_ep is not None):
            self._schedule = optim.lr_scheduler.StepLR(self._updateRule, schedule_ep, gamma=schedule_changerate, last_epoch=-1, verbose=False)
        else:
            self._schedule = None


    def update(self, sample, X, y) -> List: #sample is a list of indices, X,y are tensors of the full training set
        '''

        Parameters
        ----------
        data

        Returns
        -------

        Exception
        ---------
        AttributeError
            in case core is not set
        ValueError
            in case that data is not an numpy array
        '''
        #if self._core is None:
        #    self.error("No core is set")
        #    raise AttributeError("No core is set")

        # if not isinstance(data, List):
        #     error_text = "The argument data is not of type" + str(List) + "it is of type " + str(type(data))
        #     self.error(error_text)
        #     raise ValueError(error_text)
       
        # ## TODO: remove this stuff ---
        # example = np.asarray([record[0] for record in data])
        # label = np.asarray([record[1] for record in data])
        # self._updateRule.zero_grad()   # zero the gradient buffers
        # if self._mode == 'gpu':
        #     exampleTensor = torch.cuda.FloatTensor(example, device=self._device)
        #     if type(self._loss) is nn.MSELoss or type(self._loss) is nn.L1Loss:
        #         labelTensor = torch.cuda.FloatTensor(label, device=self._device)
        #     else:
        #         labelTensor = torch.cuda.LongTensor(label, device=self._device)
        # else:
        #     exampleTensor = torch.FloatTensor(example)
        #     if type(self._loss) is nn.MSELoss or type(self._loss) is nn.L1Loss:
        #         labelTensor = torch.FloatTensor(label)
        #     else:
        #         labelTensor = torch.LongTensor(label)
        ## ---
        self._updateRule.zero_grad()
        output = self._core(X[sample,:,:,:])
        loss = self._loss(output, y[sample])
        
        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if self._avgModel is not None:
            w_diff = torch.tensor(0., device=self._device)
            for w, w_t in zip(self._avgModel.parameters(), self._core.parameters()):
                w = w.to(device = self._device)
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += (self.mu / 2.) * w_diff
        #############################################################################
        
        loss.backward()
        self._updateRule.step()    # Does the update
        self._schedule.step()
        return [loss.data.cpu().numpy(), output.data.cpu().numpy()]

    def setParameters(self, param : PyTorchNNParameters):
        '''

        Parameters
        ----------
        param

        Returns
        -------

        Exception
        ---------
        ValueError
            in case that param is not of type Parameters
        '''

        if not isinstance(param, PyTorchNNParameters):
            error_text = "The argument param is not of type" + str(PyTorchNNParameters) + "it is of type " + str(type(param))
            self.error(error_text)
            raise ValueError(error_text)

        state_dict = OrderedDict()
        for k,v in param.get().items():
            if self._mode == 'gpu':
                if v.shape == ():
                    state_dict[k] = torch.tensor(v, device=self._device)
                else:
                    state_dict[k] = torch.cuda.FloatTensor(v, device=self._device)
            else:
                if v.shape == ():
                    state_dict[k] = torch.tensor(v)
                else:
                    state_dict[k] = torch.FloatTensor(v)
        self._core.load_state_dict(state_dict)
        self._avgModel.load_state_dict(state_dict)

    def getParameters(self) -> PyTorchNNParameters:
        '''

        Returns
        -------
        Parameters

        '''
        state_dict = OrderedDict()
        for k, v in self._core.state_dict().items():
            state_dict[k] = v.data.cpu().numpy()
        return PyTorchNNParameters(state_dict)

    def _flattenParameters(self, param):
        flatParam = []
        for k,v in param.get().items():
            flatParam += np.ravel(v).tolist()
        return np.asarray(flatParam)
        
#some utility function

def evaluateModel(model, client_idx, X_train, y_train, X_test, y_test):
    with torch.no_grad():
        train_output = model._core(X_train[client_idx])
        train_loss = model._loss(train_output, y_train[client_idx])
        _, train_predicted = torch.max(train_output, 1)
        train_total   = len(client_idx)
        train_correct = train_predicted.eq(y_train[client_idx]).sum().item()
        train_ACC = train_correct/train_total
        
        test_output = model._core(X_test)
        test_loss = model._loss(test_output, y_test)
        _, test_predicted = torch.max(test_output, 1)
        test_total   = y_test.shape[0]
        test_correct = test_predicted.eq(y_test).sum().item()
        test_ACC = test_correct/test_total
    return train_loss, train_ACC, test_loss, test_ACC

