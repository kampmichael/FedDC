import numpy

class Parameters:
    '''
    Super class of the different model- and DL library-dependent model parameter classes.
    '''

    def __init__(self):
        '''

        Implementations of this method initialize an object of 'Parameters' sub-class.

        Parameters
        ----------
        param

        Exception
        ---------
        ValueError
        '''

        raise  NotImplementedError

    def set(self):
        '''

        Implementations of this method set the model parameters.

        Parameters
        ----------
        weights

        Returns
        -------

        '''

        raise NotImplementedError

    def get(self):
        '''

        Implementations of this method return the current model parameters.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def add(self, other):
        '''

        Implementations of this method add model parameters of two models.

        Parameters
        ----------
        other: object - the model parameters to be added to the current model parameters

        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def scalarMultiply(self, scalar):
        '''

        Implementations of this method multiply the model parameters element-wise by a scalar.

        Parameters
        ----------
        scalar: float - the multiplication factor

        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def distance(self, other):
        '''

        Implementations of this method calculate the (e.g. Euclidian) distance between the current and another model in model parameter space.

        Parameters
        ----------
        other: object - the model parameters to be added to the current model parameters

        Returns
        -------
        
        '''
         
        raise NotImplementedError

    def getCopy(self):
        '''

        Implementations of this method return a copy of the current model parameters.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def toVector(self) -> numpy.array:
        '''

        Implementations of this method returns the current model parameters as a 1D numpy array.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
    def fromVector(self, v : numpy.array):
        '''

        Implementations of this method sets the current model parameters to the values given in the 1D numpy array v.

        Parameters
        ----------
        
        Returns
        -------
        
        '''
         
        raise NotImplementedError
    
