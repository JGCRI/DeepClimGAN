import numpy as np
import torch
import os
import netCDF4 as n
import torch.nn as nn
import random
from Constants import clmt_vars, scenarios, realizations
from torch.utils import data
from torch.distributions.multivariate_normal import MultivariateNormal
import logging as log


#default params
lat = 128
lon = 256
n_days = 32
n_channels = len(clmt_vars)
#number of days to look back and in the future
context_window = 5

class NETCDFDataPartition(data.Dataset):

	def __init__(self,partition,data_dir):
		"""
		Init the dataset

		param: data_pct (float) Percent of data to use for training, development and testing
		param: train_pct (float) Percent of data to use for training
		"""
		self.data = self.load_tensors(partition, data_dir)

	def __len__(self):
		"""
		Number of days (datapoints ) in the dataset
		"""
		#return self.data.shape[-1]
		return self.len
		
	def load_tensors(self, partition, data_dir):
		'''
		Load tensors for training from the partition

		param: partition (list) names of the files for current process
		param: data_dir (str) directory to load tensors from

		return tensors
		'''

		#merge tensors
		tensors = []
		self.len = 0
		for file in partition:
			location = os.path.join(data_dir, file)
			tensor = torch.load(location)
			self.len += tensor.shape[-1]
			tensors.append(tensor)
		return tensors


	def __getitem__(self, ids):
		"""
		Extracts one datapoint, which is one month of records.
		param: idx (int) batch idx

		return:
		curr_month (tensor) HxWxTxC
		avg_context(tensor) HxWx(T+context_window)x2 avg maps expanded to match the time dimension
		high_res_context (tensor) HxWxcontext_windowxC N previous days
		"""
	
		#first 5 days are reserved for the context
		start = context_window
		file_idx, idx  = ids
		
		train = self.data[file_idx]
		current_month = train[:, :, : , idx:(idx + n_days)] #output size is N_channels x H x W x 32
		high_res_context = train[:, :, :, (idx - context_window):idx]
		#get context
		keys = list(clmt_vars.keys())
		pr_idx = keys.index('pr')
		tas_idx = keys.index('tas')
		avg_p = self.expand_map(current_month[pr_idx,:,:,:], n_days, context_window)
		avg_t = self.expand_map(current_month[tas_idx,:,:,:], n_days, context_window)
		avg_context = torch.cat([avg_p, avg_t], dim=0)
		batch = {"curr_month" : current_month,
			"avg_ctxt" : avg_context,
			"high_res" : high_res_context}
		
		return batch

	def reshape_context_for_G(self, avg_context, high_res_context):
		"""
		Reshapes contexts for Generator architecture
		param: avg_context (tensor) Bx2xHxWx(T+context_window)
		param: high_res_context (tensor) BxCxHxWx5 5 previous days
		"""
		
		channels, context_length = high_res_context.shape[1], high_res_context.shape[-1]
		batch_size = avg_context.shape[0]
		lon, lat = avg_context.shape[2], avg_context.shape[3]
		high_res = high_res_context.reshape(batch_size, context_length*channels, lon, lat)
		avg_ctxt = torch.mean(avg_context, -1)
		return high_res, avg_ctxt

		
	def build_input_for_D(self, input, avg_context, high_res_context):
		"""
		Build input for discriminator, which is a concatenation of
		a map, generated by Generator and the context
		param: input (tensor) BxCxHxWxT produced by G
		param: avg_context(tensor) Bx2xHxWx(T+context_window) avg monthly precipitation and avg monthly temperature
		param: high_res_context (tensor) BxCxHxWx5 5 previous days
		
		return input for D
		"""

		last_and_curr = torch.cat([high_res_context, input], dim=4)
		input = torch.cat([last_and_curr, avg_context], dim=1)
		return input


	def get_noise(self, N, batch_size):
		"""
		Creates a multivariate normal (Gaussian) distribution
		parametrized by a mean vector and a covariance matrix
		param: N (int) Dimension of a distribution
		
		return sample from N-dimensional Gaussian distribution
		"""
		
		m = MultivariateNormal(torch.zeros(N), torch.eye(N))
		#check if need diff noise for multi-batch
		samples = []
		for i in range(batch_size):
			s = m.sample().unsqueeze_(0)
			samples.append(s)
		m = torch.cat(samples, dim=0)
		return m


	def expand_map(self, x, n_days, context_window):
		"""
		Expand average maps to concatenate with the input to D
		param: x (tensor)
		param: n_days (tensor)
		param: context_window (tensor)

		return expanded map (tensor)
		"""
		
		map = x.mean(2) #H x W x 1 x 1, we want to expand it to H x W x (T + context_window) x 1
		map = map.unsqueeze(-1).unsqueeze(-1)
		map = map.permute(3, 0, 1, 2)
		size_to_expand = n_days + context_window
		expanded_map = torch.cat(list(torch.split(map, 1, dim=3)) * size_to_expand, dim=3)
		return expanded_map

