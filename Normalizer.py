import numpy as np
import torch
from Constants import clmt_vars

class Normalizer: 

    def __init_(self, utils):

        self.clmt_vars = {
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


    def get_mean_std_for_channel(self, data, clmt_var):
        """
        Computes mean and variance for the whole dataset
        param: means (np.array) Array of all means per month
        param: variances (np.array) Array of all variances per month
        """
        mean = torch.view(data, -1).mean()
	std = torch.view(data, -1).std()
	return mean, std
	
    def get_min_max_for_channel(self, data,  clmt_var):
        """
        Computes min and max for the whole dataset
        param: means (np.array) Array of all means per month
        param: variances (np.array) Array of all variances per month
        """
        min = torch.view(data, -1).min()
        max = torch.view(data, -1).max()	
	return min, max


    def normalize_channel(self, batch, clmt_var):
        """
        Normalizes  data across the dim
        param: batch (tensor) Batch of data

        return normalized data
        """
        min = self.clmt_vars[clmt_var][0]
        max = self.clmt_vars[clmt_var][1]
        x_norm = (batch - min) / max
        return x_norm



    def denormalize(self, batch, clmt_var):
        """
        Denormalizes batch of data across the dim
        param: batch (tensor) Batch of data

        return denormalized data
        """
        min = self.clmt_vars[clmt_var][0]
        max = self.clmt_vars[clmt_var][1]
        x_denorm = batch * (max - min) + min
        return x_denorm


    def standartize_channel(self, batch, clmt_var):
        """
        Standartizes batch of data across the dim
        param: batch (tensor) Batch of data

        return normalized data
        """
        mean = self.clmt_var[clmt_var][0]
        std = self.clmt_var[clmt_var][1]
        return (batch - mean) / std

    def destandartize(self, batch):
        """
        Destandartizes batch of data across the dim
        param: batch (tensor) Batch of data

        return destandartized data
        """
        mean = self.clmt_vars[clmt_var][0]
        std = self.clmt_var[clmt_var][1]
        return mean + batch * std


    def compute_stat_vals_per_chunk(self, data):
        """
	Compute intermediate results of mean, squared sum, min and max for chunk of data
	param: data(tensor H x W x T x C)
	return mean, std, max, min
        """	
        for i, (var, val) in enumerate(clmt_vars.keys()):
                norm_type = val[1]
                if norm_type == 'stand':
                        mean, std = self.normalize(data[-1][i])
			self.clmt_vars[var] = [mean, std]
                	data[-1][i] = self.normalize_channel(data[-1][i], var)
		elif norm_type == 'norm':
                        min, max = self.standartize(data[-1][i])	
       			self.clmt_vars[var] = [min, max]
			data[-1][i] = self.standartize_channel(data[-1][i], var)
	 return data

	

    def normalize_tensor(self, tensor):
	'''
	Used to normalize the data
	param: data (tensor)
	return normalized (or standartized, depending on the channel) tensor
	'''
	for i, (var,val) in enumerate(clmt_vars.keys()):
		norm_type = val[1]
		if norm_type == 'stand':
			data[-1][i] = self.standartize(tensor[-1][i])
		elif norm_type == 'norm':
			data[-1][i] = self.normalize(tensor[-1][i])

	return data


#    def compute_stat_vals(train_len, utils, files_to_idx, dats_dir):
#
#	for i, (key, val) in enumerate(clmt_vars.items()):
#		clmt_dir = val[0]
#		file_dir = data_dir + clmt_dir
#		filenames = os.listdir(file_dir)
#		filenames = sorted(filenames)
#		#check how many files are needed per each node
#		n_files = 0
#		for j, filename in enumerate(filenames):
#			#check if the end of the file is still in the training set
#			if files_to_idx[str(j)][2] > train_len:
#				break
#			else:
#				n_files += 1
#			
#	#number of processes used
#	size = dist.get_world_size()
#	
#	#indicate how mnay files will be located on one node
#	files_partition = [n_files // size for _ in range(size)]
#	rem = n_files % size
#	i = 0
#	while rem >= 0:
#		files_partition[i] += 1
#		rem -= 1
#		i += 1
#	
#	
#	
#	for start_file_idx, end_file_idx in 
#
#	
#		#get one chunk of data for one process
#		data_chunk = utils.build_tensor(data_dir,start_file_idx, end_file_idx, n_files, train_len)
#	
#	
#		#calculate statistic for each channel
#		stat_vals_per_tensor = self.compute_stat_vals_per_chunk(data_chunk)
#	
#		#reduce this values across multiple nodes using Reduce paradigm
#		self.reduce_vals()
#	
#		#compute min, max, mean depending on the varaible for each tensor
#
#
#		#compute std, because it cannot be compute without knowing mean first
#
#
#    def reduce_vals(self):

 
