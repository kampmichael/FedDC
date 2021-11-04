from parameters import Parameters
import numpy as np
from _collections import OrderedDict

class PyTorchNNParameters(Parameters):
    '''
    Specific implementation of Parameters class for PyTorchNN learner
    Here we know that parameters are list of numpy arrays. All the methods 
    for addition, multiplication by scalar, flattening and finding distance
    are contained in this class.

    '''

    def __init__(self, stateDict : dict):
        '''
        Initialize with setting the weights values

        Parameters
        ----------
        weights - List of weights values extracted from the network with PyTorch method get_weights

        Returns
        -------
        None

        '''
        self._state = stateDict
        self._shapes = OrderedDict()
        for k in self._state:
            arr = self._state[k]
            self._shapes[k] = arr.shape

    def set(self, stateDict: dict):
        '''
        Set the weights values
        If needed to update weights inside of an existing parameters object

        Parameters
        ----------
        weights - List of weights values extracted from the network with PyTorch method get_weights

        Returns
        -------
        None

        Exception
        ---------
        ValueError
            when weights are not a list or elements of the weights list are not numpy arrays

        '''
        if not isinstance(stateDict, dict):
            raise ValueError("Weights for PyTorchNNParameters should be given as python dictionary. Instead, the type given is " + str(type(stateDict)))
            
        self._state = stateDict
        self._shapes = OrderedDict()
        for k in self._state:
            self._shapes[k] = self._state[k].shape

        # to use it inline
        return self

    def get(self) -> dict:
        '''
        Get the weights values

        Returns
        -------
        list of numpy arrays with weights values

        '''
        return self._state
    
    def add(self, other):
        '''
        Add other parameters to the current ones
        Expects that it is the same structure of the network

        Parameters
        ----------
        other - Parameters of the other network

        Returns
        -------
        None

        Exception
        ---------
        ValueError
            in case if other is not an instance of PyTorchNNParameters
            in case when the length of the list of weights is different
        Failure
            in case if any of numpy arrays in the weights list have different length

        '''
        if not isinstance(other, PyTorchNNParameters):
            error_text = "The argument other is not of type" + str(PyTorchNNParameters) + "it is of type " + str(type(other))
            self.error(error_text)
            raise ValueError(error_text)

        otherW = other.get()
        if set(self._state.keys()) != set(otherW.keys()):
            raise ValueError("Error in addition: state dictionary have different keys. This: "+str(set(self._state.keys()))+", other: "+str(set(otherW.keys()))+".")
        
        for k,v in otherW.items():
            self._state[k] = np.add(self._state[k], v)
    
    def scalarMultiply(self, scalar: float):
        '''
        Multiply weight values by the scalar

        Returns
        -------
        None

        Exception
        ---------
        ValueError
            in case when parameter scalar is not float

        '''
        if not isinstance(scalar, float):
            raise ValueError("Scalar should be float but is " + str(type(scalar)) + ".")
        
        for k in self._state:
            if isinstance(self._state[k], np.int64):
                self._state[k] *= int(scalar)
            else:
                self._state[k] = np.multiply(self._state[k], scalar, out=self._state[k], casting="unsafe")
    
    def distance(self, other) -> float:
        '''
        Calculate euclidian distance between two parameters set
        Flattens all the weights and gets simple norm of difference

        Returns
        -------
        distance between set of weights of the object and other parameters

        Exception
        ---------
        ValueError
            in case the other is not PyTorchNNParameters
            in case when length of the list of weights is different
        Failure
            in case when flattened vecrtors are different by length

        '''
        if not isinstance(other, PyTorchNNParameters):
            error_text = "The argument other is not of type" + str(PyTorchNNParameters) + "it is of type " + str(type(other))
            self.error(error_text)
            raise ValueError(error_text)

        otherW = other.get()
        if set(self._state.keys()) != set(otherW.keys()):
            raise ValueError("Error in addition: state dictionary have different keys. This: "+str(set(self._state.keys()))+", other: "+str(set(otherW.keys()))+".")
        
        w1 = self.flatten()
        w2 = other.flatten() #instead of otherW, because otherW is of type np.array instead of paramaters
        dist = np.linalg.norm(w1-w2)
        
        return dist
    
    def flatten(self) -> np.ndarray:
        '''
        Get the flattened version of weights

        Returns
        -------
        numpy array of all the layers weights flattenned and concatenated

        '''
        flatParams = []
        for k in self._state:
            flatParams += np.ravel(self._state[k]).tolist()
        return np.asarray(flatParams)
    
    def getCopy(self):
        '''
        Creating a copy of paramaters with the same weight values as in the current object

        Returns
        -------
        PyTorchNNParameters object with weights values from the current object

        '''
        newState = OrderedDict()

        for k in self._state:
            newState[k] = self._state[k].copy()
        newParams = PyTorchNNParameters(newState)
        return newParams

    def toVector(self)->np.array:
        '''

        Implementations of this method returns the current model parameters as a 1D numpy array.

        Parameters
        ----------
        
        Returns
        -------
        
        '''

        return self.flatten()

    def fromVector(self, v:np.array):
        '''

        Implementations of this method sets the current model parameters to the values given in the 1D numpy array v.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
        currPos = 0
        newState = OrderedDict()
        for k in self._shapes: #shapes contains the shapes of all weight matrices in the model and all the additional parameters, e.g., batch norm
            s = self._shapes[k]
            n = np.prod(s) #the number of elements n the curent weight matrix
            arr = v[currPos:currPos+n].reshape(s)
            newState[k] = arr.copy()
            currPos = n
        self.set(newState)
