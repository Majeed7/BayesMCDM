from abc import ABC, abstractmethod, abstractproperty

import stan
import numpy as np

import warnings
import sys
import os
import contextlib

## this is to make the STAN models work in Jupyter notebooks
import asyncio
import nest_asyncio


class MCDMProblem(ABC):
    _basic_model = ""
    _basic_model_clustering = ""
    _basic_model_sorting = ""
    
    _correlated_model = ""
    _correlated_model_clustering = ""
    _correlated_model_sorting = ""

    _is_correlated_model = False
    _is_clustering_required = False
    _is_sorting_required = False

    def __init__(self, alternatives, dm_cluster_number, alt_sort_number, num_chain, num_samples, options={}):
        self.alternatives = alternatives
        self.dm_cluster_number = dm_cluster_number
        self.alt_sort_number = alt_sort_number
        self.num_chains = num_chain
        self.num_samples = num_samples
        self.options = options

        if self.alternatives is not None or self.options.get('Sigma') is not None:
            self._is_correlated_model = True
        if self.options.get('CriteriaDependence') == False:
            self._is_correlated_model = False

        self._is_sorting_required = True if self.alt_sort_number > 0 else False
        self._is_clustering_required = True if self.dm_cluster_number > 0 else False

    @property
    def alt_no(self):
        return 0 if isinstance(self.alternatives, type(None)) else self.alternatives.shape[0]

    @abstractproperty
    def input_data(self):
        pass

    @abstractproperty
    def dm_no(self):
        pass
    
    @abstractproperty
    def criteria_no(self):
        pass
    
    @abstractmethod
    def _check_input_data(self):
        pass

    @property
    def model(self):
        model = 'self._'

        model = model + 'correlated_model' if self._is_correlated_model else model + 'basic_model'
        model = model + '_clustering' if self._is_clustering_required else model
        model = model + '_sorting' if self._is_sorting_required else model

        self.model_name = model
        # print("The used model is: ", model)
        return eval(model)

    def _get_common_data(self):
        data = {}
        data['gamma_param'] = 0.01
        data['sigma_coef'] = 0.1
        data['DmNo'] = self.dm_no
        data['CNo'] = self.criteria_no
        
        if self.dm_cluster_number > 0:
            data['DmC'] = self.dm_cluster_number

        if not isinstance(self.alternatives, type(None)):
            data['Alt'] = self.alternatives
            data['AltNo'] = self.alt_no

        if self.alt_sort_number > 0:
            data['AltC'] = self.alt_sort_number
            data['eAlt'] = np.ones(self.alt_sort_number)

        if self._is_correlated_model:
            data['mu'] = 0.01 * np.ones(self.criteria_no)
            data['Sigma'] = np.cov(self.alternatives.T)
            if not isinstance(self.options.get('Sigma'), type(None)):
                data['Sigma'] = self.options.get('Sigma')
                assert data['Sigma'].shape == (self.criteria_no, self.criteria_no)

        if self.alt_sort_number > 0 and isinstance(self.alternatives, type(None)):
            raise Exception("Alternatives should be given as input for the sorting problem!")

        return data

    def sampling(self):
        if self._check_input_data:
            nest_asyncio.apply()
            asyncio.run(asyncio.sleep(1))
            with self.suppress_stan_warnings():
                posterior = stan.build(self.model, data=self.input_data, random_seed=1)
                
                print("Sampling started...")
                self.samples = posterior.sample(num_chains=self.num_chains, num_samples=self.num_samples, num_warmup=300)
                print("Sampling finished.")
                self.process_samples() 
        else:
            raise Exception("The input data is not valid")

    @contextlib.contextmanager
    def suppress_stan_warnings(self):
        import io
        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            yield


    def process_samples(self):
        if self.__class__.__name__ != "StandardWeightAnalyzer":
            self.dm_weight_samples = self.samples['W'].T
            self.dm_weight = np.mean(self.dm_weight_samples, axis=0)


        if self._is_clustering_required:
            self.cluster_center_samples = self.samples['wc']
            self.cluster_centers = np.mean(self.cluster_center_samples, axis=2)
            self.dm_membership_samples = self.samples['theta']
            self.dm_membership = np.mean(self.dm_membership_samples, axis=2)
            
        elif self._is_sorting_required:
            self.aggregated_weight_samples = self.samples['wStar']
            self.aggregated_weight = np.mean(self.aggregated_weight_samples, axis=1)
            
            soft_z_un = np.mean(self.samples['soft_z'], axis=2)
            soft_z = np.exp(soft_z_un)
            sum_soft_z = np.sum(soft_z, axis=1).reshape((self.alt_no, 1))
            self.alternative_membership = np.divide(soft_z, sum_soft_z)
            self.alternative_sorting = np.argmax(soft_z, axis=1)

            self.alternative_values = 1 / (1 + np.exp(-self.samples['v']))

            mu_un = np.mean(self.samples['altMu'], axis=1)
            self.sorting_centers = 1 / (1 + np.exp(-mu_un))
        
        else:
            self.aggregated_weight_samples = self.samples['wStar'].T
            self.aggregated_weight = np.mean(self.aggregated_weight_samples, axis=0)
