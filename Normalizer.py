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

	

	def log_denormalize_channel(self, data):
		"""
		Log denormalize channel
		"""
		x = np.exp(data)- 1
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
	

	def destandartize_channel(self, batch, clmt_var):
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
				tsr[i] = self.standartize_channel(tsr[i], var)
		return tsr



	def denormalize(self, tsr):
		"""
		Denormalize the tensor
		"""
	
		#tsr = tsr.numpy()
		for i, (var, val) in enumerate(clmt_vars.items()):
			norm_type = val[1]
			if norm_type == 'log_norm':
				tsr[i] = self.log_denormalize_channel(tsr[i])
			elif norm_type == 'stand':
				tsr[i] = self.destandartize_channel(tsr[i], var)
		return tsr
	


	def load_means_and_stds(self, dir):
		"""
		Load means and stds
		"""		

		tas_mean, tas_std = torch.load(dir+'tas_mean.pt').item(), torch.load(dir+'tas_std.pt').item()
		tasmin_mean, tasmin_std = torch.load(dir+'tasmin_mean.pt').item(), torch.load(dir+'tasmin_std.pt').item()
		tasmax_mean, tasmax_std = torch.load(dir+'tasmax_mean.pt').item(), torch.load(dir+'tasmax_std.pt').item()
		rhs_mean, rhs_std = torch.load(dir+'rhs_mean.pt').item(), torch.load(dir+'rhs_std.pt').item()
		rhsmin_mean, rhsmin_std = torch.load(dir+'rhsmin_mean.pt').item(), torch.load(dir+'rhsmin_std.pt').item()
		rhsmax_mean, rhsmax_std = torch.load(dir+'rhsmax_mean.pt').item(), torch.load(dir+'rhsmax_std.pt').item()


		"""
		print(tas_mean, tas_std)
		print(tasmin_mean, tasmin_std)
		print(tasmax_mean, tasmax_std)
		print(rhs_mean, rhs_std)
		print(rhsmin_mean, rhsmin_std)
		print(rhsmax_mean, rhsmax_std)
		"""

				
		self.clmt_stats['tas'] = [tas_mean, tas_std]
		self.clmt_stats['tasmin'] = [tasmin_mean, tasmin_std]
		self.clmt_stats['tasmax'] = [tasmax_mean, tasmax_std]
		self.clmt_stats['rhs'] = [rhs_mean, rhs_std]
		self.clmt_stats['rhsmin'] = [rhsmin_mean, rhsmin_std]
		self.clmt_stats['rhsmax'] = [rhsmax_mean, rhsmax_std]
		
		return		


