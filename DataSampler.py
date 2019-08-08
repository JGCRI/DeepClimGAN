from torch.utils.data.sampler import Sampler
import torch
import numpy as np


class DataSampler(Sampler):
	"""
	Samples elements according to DataSampler
	Arguments:
		data_source (DataSet): dataset to sample from
	"""


	def __init__(self, batch_size, data, context_window, n_days):
		"""
		Initialize sampler
		"""

		self.n_files = len(data)
		self.file_indices = []
		
		idx_start = context_window
		self.batch_size = batch_size
		self.total_len = 0
		for i in range(n_files):
			data_len = data[i].shape[-1]
			idx_end = data_len - n_days - 1
			self.idx_range = [idx_start, idx_end]
			indices = self.get_indices(idx_range)
			self.total_len += len(indices)
			file_indices.append(indices)	


			
	def get_indices(self, idx_range):
		"""
		Create indices of datapoints to sample from the dataset.
		Don't use first n days, which are reserved for the context
		Also don't use last n_days, since if we sample one month, we will 
			get out of bound
		"""
		
		#count number of batches using formula: a_0 + d(n - 1) <= a_n, 
		#where a_0 = ontext_window, d = batch_size, n = n_batches, a_n = idx_end
		
		lb, ub = idx_range[0], idx_range[1]
		batch_size = self.batch_size
		n_batches = (ub - lb) // batch_size + 1
		indices = [i for i in range(lb, ub + 1)]
		return indices		
	

	def permute(self):
		"""
		Shuffle indices after each epoch
		"""
		files = self.file_indices
		#permute indices within a file
		for indices in files:
			indices = np.random.permutation(indices)
		

	def __iter__(self):
		"""
		Iterator itself
		"""
		n_files = self.n_files
		#i is a file in the partition, j is the index
		for i in range(n_files):
			for j in range(len(n_files[i]):
				return i, j
		#return (self.indices[i] for i in range(len(self.indices)))
				

	def __len__(self):
		"""
		Get the length of all the valid indices
		"""
		#return len(self.indices)
		return self.total_len
		
