########
Examples
########

.. _example_model_selection_aic:

Model selection via AIC
=======================

::

    from pyPhenology import utils
    import numpy as np

    models_to_test = ['ThermalTime','Alternating','Linear']

    observations, temp = utils.load_test_data(name='vaccinium')

    # Only keep the leaf phenophase
    observations = observations[observations.phenophase==371]

    observations_test = observations[0:10]
    observations_train = observations[10:]

    # AIC based off mean sum of squares
    def aic(obs, pred, n_param):
        return len(obs) * np.log(np.mean((obs - pred)**2)) + 2*(n_param + 1)

    best_aic=np.inf
    best_base_model = None
    best_base_model_name = None

    for model_name in models_to_test:
        Model = utils.load_model(model_name)
        model = Model()
        model.fit(observations_train, temp, optimizer_params='practical')
        
        model_aic = aic(obs = observations_test.doy.values,
                        pred = model.predict(observations_test,
                                             temp),
                        n_param = len(model.get_params()))
        
        if model_aic < best_aic:
            best_model = model
            best_model_name = model_name
            best_aic = model_aic
            
        print('model {m} got an aic of {a}'.format(m=model_name,a=model_aic))
        
    print('Best model: {m}'.format(m=best_model_name))
    print('Best model paramters:')
    print(best_model.get_params())


.. _example_bootstrap_rmse:

==============================
Model RMSE Using Bootstrapping
==============================

.. ipython:: python

    from pyPhenology import utils, models
    import numpy as np
    import pandas as pd
    #import matplotlib.pyplot as plt
    
    observations, temp = utils.load_test_data(name='vaccinium')
    
    datasets_to_use = ['vaccinium','aspen']
    phenophases_to_use = ['budburst','flowers']
    
    num_boostraps=5
    
    # Two Thermal Time models with fixed start day of Jan 1, and 
    # with different fixed temperature thresholds.
    # Each getting variation using 50 bootstraps.
    bootstrapped_tt_model_1 = models.BootstrapModel(core_model=models.ThermalTime,
                                                    num_bootstraps=num_boostraps,
                                                    parameters={'t1':1,
                                                                'T':0})
    
    bootstrapped_tt_model_2 = models.BootstrapModel(core_model=models.ThermalTime,
                                                    num_bootstraps=num_boostraps,
                                                    parameters={'t1':1,
                                                                'T':5})
    
    models_to_fit = {'TT Temp 0':bootstrapped_tt_model_1,
                     'TT Temp 5':bootstrapped_tt_model_2}
    
    results = pd.DataFrame()
    
    for dataset in datasets_to_use:
        for phenophase in phenophases_to_use:
            
            observations, temp = utils.load_test_data(name=dataset,
                                                      phenophase=phenophase)
            
            # Setup 20% train test split using pandas methods
            observations_test = observations.sample(frac=0.2,
                                                    random_state=1)
            observations_train = observations[~observations.index.isin(observations_test.index)]
            
            observed_doy = observations_test.doy.values
            
            for model_name, model in models_to_fit.items():
                model.fit(observations_train, temp, optimizer_params='testing')
                
                # Using aggregation='none' in BoostrapModel predict
                # returns results for all bootstrapped models in an
                # (num_bootstraps, n_samples) array. This will calculate
                # the RMSE of each model and var variation around that. 
                predicted_doy = model.predict(observations_test, temp, aggregation='none')
    
                rmse = np.sqrt(np.mean( (predicted_doy - observed_doy)**2, axis=1))
                
                results_this_set = pd.DataFrame()
                results_this_set['rmse'] = rmse
                results_this_set['dataset'] = dataset
                results_this_set['phenophase'] = phenophase
                results_this_set['model'] = model_name
    
                results = results.append(results_this_set, ignore_index=True)
    
    @savefig example_rmse_boxplot.png width=4in
    bp = results.boxplot(column='rmse', by=['dataset','phenophase','model'])

