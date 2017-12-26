import numpy as np
import pandas as pd
from . import utils
from .base import _base_model


class Sequential(_base_model):
    """The sequentail model using a triangular response for
    chilling and growing degree days for forcing.
    
    Parameters
    ----------
    t0 : int
        The doy which chilling accumulation beings
    
    T : int
        The threshold above which forcing accumulates
    
    F : int, > 0
        The total forcing units required
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'t0':(-67,298),'c_t_min':(-25,25),'c_t_opt':(0,1000),
                                        'c_t_max':(0,10), 'C':(0,100), 'f_t':(0,20), 'F':{0,1000}}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t0, c_t_min, c_t_opt, c_t_max,
                                                     C, f_t, F):
        chill_days = utils.traingle_response(temperature, t_min=c_t_min,
                                             t_opt = c_t_opt, t_max=c_t_max)
        chill_days[doy_series<t0]=0
        chill_days = utils.forcing_accumulator(chill_days)
        
        t1_values = utils.doy_estimator(forcing = chill_days,
                                        doy_series=doy_series,
                                        threshold=C)
        
        
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)