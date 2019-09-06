from torch.utils.data.sampler import Sampler
import torch
import numpy as np
import logging as log

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

		self.data = data
		self.n_files = len(data)
		self.file_indices = []
		
		idx_start = context_window
		self.batch_size = batch_size
		self.total_len = 0
		
		for i in range(self.n_files):
			data_len = data[i].shape[-1]
			idx_end = data_len - n_days - 1
			self.idx_range = [idx_start, idx_end]
			indices = self.get_indices(self.idx_range)
			self.total_len += len(indices)
			self.file_indices.append(indices)	

		self.file_indices = np.asarray(self.file_indices)
		self.permute()

		
			
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
		indices = np.asarray(indices)
		
		return indices		
	

	def permute(self):
		"""
		Shuffle indices after each epoch
		"""
		files = self.file_indices
		#permute indices within a file
		log.info("Permuting indices in each file")
		for i in range(self.n_files):
			self.file_indices[i] = np.random.permutation(self.file_indices[i])


	def __iter__(self):
		"""
		Iterator itself
		"""
		n_files = self.n_files
		file_indices = self.file_indices
		
		for i in range(n_files):
			file = file_indices[i]
			for j in range(len(file)):
				yield (i, file[j])	
				

	def __len__(self):
		"""
		Get the length of all the valid indices
		"""
		

		return self.total_len
		
