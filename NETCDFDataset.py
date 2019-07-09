import numpy as np
import torch
import os
import netCDF4 as n
import torch.nn as nn
import random
from Constants import clmt_vars
from torch.utils import data
#use to visualize the data
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from torch.distributions.multivariate_normal import MultivariateNormal



#default params
lat = 128
lon = 256
n_days = 32
n_channels = len(clmt_vars)
#number of days to look back and in the future
context_window = 5

data_dir = '../clmt_data/'

class NETCDFDataset(data.Dataset):

	def __init__(self, data_dir, data_pct, train_pct):
		"""
		Init the dataset

		param: data_pct (float) Percent of data to use for training, development and testing
		param: train_pct (float) Percent of data to use for training
		"""

		self.data, self.len  = self.build_dataset(data_dir, data_pct)
		self.train_len, self.dev_len, self.test_len = self.create_split_bounds(self.len, train_pct)
		#will be initialized later
		self.normalized_train = None
		self.normalized_dev = None


	def __len__(self):
		"""
		Number of days (datapoints ) in the dataset
		"""
		return self.len

	def __getitem__(self, idx):
		"""
		Extracts one datapoint, which is one month of records.
		param: idx (int) batch idx

		return:
		curr_month (tensor) HxWxTxC
		avg_context(tensor) HxWx(T+context_window)x2 avg maps expanded to match the time dimension
		high_res_context (tensor) HxWxcontext_windowxC N previous days
		"""
		train = self.normalized_train
		#first 5 days are reserved for the context
		start = context_window
#		if start + n_days + idx == self.train_len:
#			return None
		current_month = train[:, :, : , (start + idx):(start + n_days + idx)]#output size is N_channels x H x W x 32
		high_res_context = train[:, :, :, idx:(start + idx)]
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
		

	def export_netcdf(self,filename, var_name):
		"""
		Export data from one .nc file
		:param filename: (str) Name of the .nc file
		:param var_name: (str) Name of the climate variable
		:return exported from .nc file variable
		"""

		nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')
		var = nc.variables[var_name][:]
		return var



	def build_dataset(self, data_dir, data_pct):
		"""
		Builds dataset out of all climate variables of shape HxWxNxC, where:
		N - #of datapoints (days)
		H - latitude
		W - longitude
		C - #of channels (climate variables)
		param: data_dir(str) directory with the data
		param: data_pct (float) percent of the data to use
		:return dataset as a tensor
		"""

		#all tensors is a list of tensor, where each tensor has data about one climate variable
		all_tensors = []
		count_files = 0
		for i, (key, val) in enumerate(clmt_vars.items()):
			clmt_dir = val[0]
			file_dir = data_dir + clmt_dir
			#create tensor for one climate variable
			tensors_per_clmt_var = []
			#sort files in ascending order (based on the date)
			filenames = os.listdir(file_dir)
			filenames = sorted(filenames)
			count_files += len(filenames)
			for j, filename in enumerate(filenames):
			        raw_clmt_data = self.export_netcdf(file_dir + filename,key)
			        raw_tsr= torch.tensor(raw_clmt_data, dtype=torch.float32)
			        tensors_per_clmt_var.append(raw_tsr)

			#concatenate tensors along the size dimension
			concat_tsr = torch.cat(tensors_per_clmt_var, dim=0)
			all_tensors.append(concat_tsr)

			print("Finished parsing {} files for variable \"{}\" ".format(len(filenames),key))
		res_tsr = torch.stack(all_tensors, dim=3)
		print("Finished parsing {} files total".format(count_files))
		tensor_len = res_tsr.shape[0]

		#if we decided not to use all the data;add window size (5 days) that are used for context
		data_len = int(np.floor(tensor_len * data_pct) + context_window)
		res_tsr = res_tsr[:data_len, :, :, :]


		res_tsr = res_tsr.permute(3, 1, 2, 0)#C x H x W x N
		print("The size of result tensor is {}".format(res_tsr.shape))
		return res_tsr, data_len
	

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
		"""
		last_and_curr = torch.cat([high_res_context, input], dim=4)
		input = torch.cat([last_and_curr, avg_context], dim=1)
		return input


	def create_split_bounds(self, N, train_pct):
		#reference: https://github.com/hutchresearch/ml_climate17/blob/master/resnet/utils/DataHelper.py
		"""
		Computes split bounds for train, dev and test.
		:param N (int) Length of the dataset
		:param train_pct=0.7 (float) Percent of data to use for training
		:return: train (default to 70% of all data)
		:return: dev (default to 15% of all data)
		:return: test (default to 15% of all data)
		"""

		train_len = int(round(train_pct * N))
		if (N - train_len) % 2 == 1:
		        train_len += 1

		#NOTE: We assume that dev_len = test_len
		dev_len = test_len = int((N - train_len) / 2)

		assert "Not all data points are being used. Check create_split_bounds()", \
		        (train_len + dev_len + test_len) == N


		return train_len, dev_len, test_len



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
		#m = m.sample().unsqueeze_(0).repeat(batch_size,1)
		return m


	def expand_map(self, x, n_days, context_window):
		"""
		Expand average maps to concatenate with the input to D
		param: x (tensor)
		param: n_days (tensor)
		param: context_window (tensor)

		return expanded map (tensor)
		"""
		map = x.mean(2)#H x W x 1 x 1, we want to expand it to H x W x (T + context_window) x 1
		map = map.unsqueeze(-1).unsqueeze(-1)
		map = map.permute(3, 0, 1, 2)
		size_to_expand = n_days + context_window
		expanded_map = torch.cat(list(torch.split(map, 1, dim=3)) * size_to_expand, dim=3)
		return expanded_map


# def visualize_channels(data):
# 	"""
# 	Plot the distribution of the data
# 	"""
# 	for i, (var, val) in enumerate(clmt_vars.keys()):
# 		visualize_tensor(np.take(data, i, axis=data.shape[-1]), var_name)

#
# def visualize_channel(tensor, var_name):
# 	"""
# 	Plot data distribution for one channel
# 	param: tensor: H x W x T
# 	"""
# 	flattened = tensor.flatten()
# 	plot = sns.distplot(x, kde=True, rug=False)
# 	fig = plot.get_figure()
# 	fig.savefig('../' + var_name + '.png')

