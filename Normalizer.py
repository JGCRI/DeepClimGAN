import numpy as np
import torch
from Constants import clmt_vars

class Normalizer:

    def __init__(self):
        self.clmt_stats = {
            #standartize tmp
            "tas_day" : [0, 0],#mean, std
            #normalize pr
            "pr" : [0, 0],#min, max
            #normalize relative humidity
            "rhs" : [0, 0],
            #normalize humidity
            "tas_max" : [0, 0],
            #normalize humidity
            "tax_min" : [0, 0]
        }


    def get_mean_std_for_channel(self, data):
        """
        """

        mean = torch.mean(data).item()
        std = torch.std(data).item()
        return mean, std

    def get_min_max_for_channel(self, data):
        """
        """
        min = torch.min(data)
        max = torch.max(data)
        return min, max


    def normalize_channel(self, data, clmt_var):
        """
        Normalizes  data across the dim
        param: data (tensor) H x W x T
        return normalized data
        """
        min = self.clmt_stats[clmt_var][0]
        max = self.clmt_stats[clmt_var][1]
        x_norm = (data - min) / max
        return x_norm


    def log_normalize_channel(self, data, clmt_var):
        """
	Log normalize the data (base e)
	param: data (tensr) H x W x T
	return normalized data
        """
        x = np.log(1 + x)
        return x


    def denormalize(self, batch, clmt_var):
        """
        Denormalizes batch of data across the dim
        param: batch (tensor) Batch of data
        return denormalized data
        """
        min = self.clmt_stats[clmt_var][0]
        max = self.clmt_stats[clmt_var][1]
        x_denorm = batch * (max - min) + min
        return x_denorm


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


    def normalize(self, ds, dataset_type):
        """
	Normalizes the data
	param: data (tensor)
	param: dataset_type(str) type of a dataset
	return normalized data
        """
        if dataset_type not in ['train', 'dev']:
                raise Exception('Correct dataset type is not provided. Use \'train\' or \'dev\' ')

        up, lb = 0, 0
        if dataset_type == 'train':
                lb = ds.train_len
        elif dataset_type == 'dev':
                ub = ds.train_len
                lb = ds.train_len + ds.dev_len

        data = ds.data[:,:,:,up:lb]

        for i, (var, val) in enumerate(clmt_vars.items()):
                norm_type = val[1]
                if norm_type == 'stand':
                        mean, std = self.get_mean_std_for_channel(data[i])
                        self.clmt_stats[var] = [mean, std]
                        data[i] = self.normalize_channel(data[i], var)
                elif norm_type == 'norm':
                        min, max = self.get_min_max_for_channel(data[i])
                        self.clmt_stats[var] = [min, max]
                        data[i] = self.standartize_channel(data[i], var)
        return data

