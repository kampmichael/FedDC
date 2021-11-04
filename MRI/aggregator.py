from typing import List
from parameters import Parameters

class Aggregator():


    def __init__(self, name = "Aggregator"):
        self.name = name

    def __call__(self, params : List[Parameters]) -> Parameters:
        '''
        Aggregator call method, combines Parameters into one model's Parameters
        Specific implementation is different for different approaches.

        Parameters
        ----------
        params - list with Parameters of models to be aggregated

        Returns
        -------
        Parameters object for the aggregated model

        '''

        raise NotImplementedError
        