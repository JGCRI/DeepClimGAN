import os
import netCDF4 as n
import numpy as np
import torch.nn as nn
import torch
import random
from Constants import clmt_vars

#default params
lat = 128
lon = 256
time = 30
n_channels = len(clmt_vars)

class DataLoader:

	def __init__(self):
		self.splitted_tensor = None
		self.train = None
		self.dev = None
		self.test = None

	def export_netcdf(self,filename, var_name):
		"""
		Export data from one .nc file 
		
		:param filename: (str) Name of the .nc file
		:param var_name: (str) Name of the climate variable

		:return exported from .nc file variable	
		"""
		
		nc = n.Dataset(filename, 'r', format='NETCDF4_CLASSIC')
		lon = nc.variables['lon'][:]
		lat = nc.variables['lat'][:]
		var = nc.variables[var_name][:]	
		return var


	def build_dataset(self, data_dir):
		"""
		Builds dataset out of all climate variables of shape NxHxWxTxC, where:
			N - #of datapoints (days)
			H - latitude
			W - longitude
			T - time (30days)
			C - #of channels (climate variables)

		:return dataset as a tensor
		"""
				
		#all tensors is a list of tensor, where each tensor has data about one climate variable	
		all_tensors = []
		count_files = 0
		for i, (key, val) in enumerate(clmt_vars.items()):
			file_dir = data_dir + val
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

		#permuate tensor for convenience
		res_tsr = res_tsr.permute(1, 2, 0, 3)
		
	
		#split tensor into chunks, each chunk 30 days long
		n_chunks = res_tsr.shape[2] // time
		splitted_tsrs = torch.chunk(res_tsr, n_chunks, dim=2)
	
		#concatenate tensors back
		splitted_tsr = torch.stack(splitted_tsrs, dim=0)
		print("The size of result tensor is {}".format(splitted_tsr.shape))
		self.splitted_tensor = splitted_tsr

		return self.splitted_tensor


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

		

	def split_dataset(self, data, train_pct=0.7):
		"""
		reference: https://github.com/hutchresearch/ml_climate17/blob/master/resnet/utils/DataHelper.py
		Splits data into train, dev and test sets.
		Since we are using unsupervised training, no labels are provided

		:param data (tensor) Dataset as a tensor
		:param train_pct (float) Percent of data to be used for the training set		

		:return train_x, dev_x, test_x
		"""
		
		train_len, dev_len, test_len = self.create_split_bounds(data.shape[0], train_pct)
		
		#Train 70%
		train = data[0:train_len]

		#Dev 15%
		dev_ub = (train_len + dev_len)
		dev = data[train_len:dev_ub]

		#Test
		test = data[dev_ub:]

		assert "One of the sets contains an unexpected number of elements", \
			(train.shape[0] == train_len and dev.shape[0] == dev_len and test.shape[0] == test_len)

		self.train, self.dev, self.test = train, dev, test
		
		return train, dev, test


	
	def get_batch(self, data, batch_size, shuffle=True):
		""""
		
		 Returns batch of training data
		:param data: data (N-d tensor)
		:param batch_size (int) Size of a mini-batch
		
		return a batch of data of shape: batch_size x H x W x T x C	
		"""
		n_batches = 32
		data_len = data.shape[0]
		n_batches = data_len // batch_size #NOTE: last batch may be partial
		

		batches = torch.chunk(data, n_batches, dim=0)

                #concatenate tensors back to get shape batch_size x H x W x T x C
                data = torch.stack(batches, dim=0)		
		print(data.shape)

		#if shuffle:
			
			
		#rand_idx = random.int(data_len - batch_size)
		
		#return data[train_idx:batch_size,:]			
		
		
		
		
		
	def get_noise(self, N):
		"""
		Creates a multivariate normal (Gaussian) distribution 
		parametrized by a mean vector and a covariance matrix

		param: N (int) Dimension of a distribution
		
		return sample from N-dimensional Gaussian distribution 
		"""
		m = MultivariateNormal(torch.zeros(N), torch.eye(N))		
		m.sample()
		return m
