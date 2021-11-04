from parameters_pytorch import PyTorchNNParameters

import numpy as np
from typing import List
import torch
import time
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class PyTorchNN():
    def __init__(self, batchSize, mode, device, name = "PyTorchNN"):
        self._core          		= None
        self._flattenReferenceParams    = None
        self._mode			= mode
        self._device			= device
        self.name = name

    def setCore(self, network):
        self._core = network

    def setModel(self, param: PyTorchNNParameters, setReference: bool):
        super(PyTorchNN, self).setModel(param, setReference)
        
        if setReference:
            self._flattenReferenceParams = self._flattenParameters(param)

    def setLoss(self, lossFunction):
        self._loss = eval("nn." + lossFunction + "()")

    def setUpdateRule(self, updateRule, learningRate, schedule_ep, schedule_changerate, **kwargs):
        additional_params = ""
        for k in kwargs:
            additional_params += ", " + k  + "=" + str(kwargs.get(k))
        self._updateRule = eval("optim." + updateRule + "(self._core.parameters(), lr=" + str(learningRate) + additional_params + ")")
        if (schedule_ep is not None):
            self._schedule = optim.lr_scheduler.StepLR(self._updateRule, schedule_ep, gamma=schedule_changerate, last_epoch=-1)
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
        if self._core is None:
            self.error("No core is set")
            raise AttributeError("No core is set")

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
        loss.backward()
        self._updateRule.step()    # Does the update
        if self._schedule is not None:
            self._schedule.step()
        #return [loss.data.cpu().numpy(), output.data.cpu().numpy()]
        return

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
        #train_predicted = torch.round(torch.sigmoid(train_output))
        #train_predicted = torch.argmax(train_output)
        _, train_predicted = torch.max(train_output, 1)

        train_total   = len(client_idx)
        train_correct = train_predicted.eq(y_train[client_idx]).sum().item()
        train_ACC = train_correct/train_total
         
        test_output = model._core(X_test)
        test_loss = model._loss(test_output, y_test)
        #test_predicted = torch.round(torch.sigmoid(test_output))
        _, test_predicted = torch.max(test_output, 1)
        #test_predicted = torch.argmax(test_output)

        test_total   = y_test.shape[0]
        test_correct = test_predicted.eq(y_test).sum().item()
        test_ACC = test_correct/test_total
    return train_loss, train_ACC, test_loss, test_ACC

