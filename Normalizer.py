import numpy as np
import torch
from Constants import clmt_vars

class Normalizer:

    def __init__(self):
        self.clmt_stats = {}


    def log_normalize_channel(self, data):
        """
	Log normalize the data (base e)
	param: data (tensr) H x W x T
	return normalized data
        """
	
        x = np.log(1 + data)
        return x


    def standartize_channel(self, batch, clmt_var):
        """
        Standartizes batch of data across the dim
        param: batch (tensor) Batch of data
        return normalized data
        """
        mean = self.clmt_stats[clmt_var][0]
        std = self.clmt_stats[clmt_var][1]
        return (batch - mean) / std

    def destandartize(self, batch):
        """
        Destandartizes batch of data across the dim
        param: batch (tensor) Batch of data
        return destandartized data
        """
        mean = self.clmt_stats[clmt_var][0]
        std = self.clmt_stats[clmt_var][1]
        return mean + batch * std


    def normalize(self, tsr):
        """
	Normalizes the data
	param: data (tensor)
	param: dataset_type(str) type of a dataset
	return normalized data
        """

        for i, (var, val) in enumerate(clmt_vars.items()):
                norm_type = val[1]
                if norm_type == 'log_norm':
                        tsr[i] = self.log_normalize_channel(tsr[i])
                elif norm_type == 'stand':
                        self.clmt_stats[var] = [mean, std]
                        tsr[i] = self.standartize_channel(tsr[i], var)
        return data

