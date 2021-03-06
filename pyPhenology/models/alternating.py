import numpy as np
from . import utils
from .base import _base_model

class Alternating(_base_model):
    """Alternating model.
    
    Originally defined in Cannell & Smith 1983.
    
    Phenological event happens the first day that forcing is greater 
    than an exponential curve of number of chill days.
    
    .. math::
        \\sum_{t=t1}^{DOY}R_{f}(T_{i})\\geq a + be^{cNCD(t)}
    
    where:
    
    .. math::
        R_{f}(T_{i}) = max(T_{i}-threshold, 0)
    
    Parameters:
        a : int | float
            | :math:`a` - Intercept of chill day curve
            | default : (-1000,1000)
        
        b : int | float, > 0
            | :math:`b` - Slope of chill day curve
            | default : (0,5000)
        
        c : int | float, < 0
            | :math:`c` - scale parameter of chill day curve
            | default : (-5,0)
            
        threshold : int | flaot
            | :math:`threshold` - Degree threshold above which forcing accumulates, and below which chilling accumulates.
            | default : 5
            
        t1 : int
            | :math:`` - DOY which forcing and chilling accumulationg starts.
            | default : 1 (Jan 1)

    Notes:
        Cannell, M. G. R., & Smith, R. I. (1983). 
        Thermal Time, Chill Days and Prediction of Budburst in Picea sitchensis.
        The Journal of Applied Ecology, 20(3), 951.
        https://doi.org/10.2307/2403139

    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'threshold':5, 't1':1,
                                        'a':(-1000,1000), 'b':(0,5000), 'c':(-5,0)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, a, b, c, threshold, t1):
        chill_days = ((temperature < threshold)*1).copy()
        chill_days[doy_series < t1]=0
        chill_days = utils.forcing_accumulator(chill_days)

        # Accumulated growing degree days from Jan 1
        gdd = temperature.copy()
        gdd[gdd < threshold]=0
        gdd[doy_series < t1]=0
        gdd = utils.forcing_accumulator(gdd)

        # Phenological event happens the first day gdd is > chill_day curve
        chill_day_curve = a + b * np.exp( c * chill_days)
        difference = gdd - chill_day_curve

        # The estimate is equal to the first day that
        # gdd - chill_day_curve > 0
        return utils.doy_estimator(difference, doy_series, threshold=0)
    
class MSB(_base_model):
    """Macroscale Species-specific Budburst model.
    
    Extension of the Alternating model which adds a correction (:math:`d`)
    using the mean spring temperature.
    
    .. math::
        \sum_{t=t1}^{DOY}R_{f}(T_{i})\geq a + be^{cNCD_{i}} +dT_{mean}
        
    where:
    
    .. math::
        R_{f}(T_{i}) = max(T_{i}-threshold, 0)
        
    Spring is DOY 1-60 as described in Jeong et al. 2013.
    
    Parameters:
        a : int | float
            | :math:`a` - Intercept of chill day curve
            | default : (-1000,1000)
        
        b : int | float, > 0
            | :math:`b` - Slope of chill day curve
            | default : (0,5000)
        
        c : int | float, < 0
            | :math:`c` - scale parameter of chill day curve
            | default : (-5,0)
        
        d : int | float
            | :math:`d` - Correction factor using spring temperature
            
        threshold : int | flaot
            | :math:`threshold` - Degree threshold above which forcing 
              accumulates, and below which chilling accumulates.
            | default : 5
            
        t1 : int
            | :math:`` - DOY which forcing and chilling accumulationg starts.
            | default : 1 (Jan 1)
    
    Notes:
        Jeong, S.-J., Medvigy, D., Shevliakova, E., & Malyshev, S. (2013). 
        Predicting changes in temperate forest budburst using continental-scale observations and models.
        Geophysical Research Letters, 40(2), 359–364.
        https://doi.org/10.1029/2012Gl054431

    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'threshold':5, 't1':1, 'd':(-100,100),
                                        'a':(-1000,1000), 'b':(0,5000), 'c':(-5,0)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, a, b, c, d, threshold, t1):
        chill_days = ((temperature < threshold)*1).copy()
        chill_days[doy_series < t1]=0
        chill_days = utils.forcing_accumulator(chill_days)

        # Accumulated growing degree days from Jan 1
        gdd = temperature.copy()
        gdd[gdd < threshold]=0
        gdd[doy_series < t1]=0
        gdd = utils.forcing_accumulator(gdd)

        chill_day_curve = a + b * np.exp( c * chill_days)
        
        # Make the spring temps the same shape as chill_day_curve
        # for easy addition.
        mean_spring_temp = utils.mean_temperature(temperature, doy_series,
                                                  start_doy=1, end_doy=60)
        mean_spring_temp *= d
        # Add in correction based on per site spring temperature
        chill_day_curve += mean_spring_temp

        # Phenological event happens the first day gdd is > chill_day curve
        difference = gdd - chill_day_curve

        # The estimate is equal to the first day that
        # gdd - chill_day_curve > 0
        return utils.doy_estimator(difference, doy_series, threshold=0)
