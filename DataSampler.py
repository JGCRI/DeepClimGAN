from torch.utils.data.sampler import Sampler
import torch
import numpy as np


class DataSampler(Sampler):
	"""
	Samples elements according to DataSampler
	Arguments:
		data_source (DataSet): dataset to sample from
	"""


	def __init__(self,data_source, batch_size, context_window, n_days):
		"""
		Initialize sampler
		"""
		self.train_data = data_source.normalized_train
		self.batch_size = batch_size
		idx_start = context_window
		idx_end = self.train_data.shape[-1] - n_days - 1
		self.idx_range = [idx_start, idx_end]
		self.indices = self.get_indices()

			
	def get_indices(self):
		"""
		Create indices of datapoints to sample from the dataset.
		Don't use first n days, which are reserved for the context
		Also don't use last n_days, since if we sample one month, we will 
			get out of bound
		"""
		
		#count number of batches using formula: a_0 + d(n - 1) <= a_n, 
		#where a_0 = ontext_window, d = batch_size, n = n_batches, a_n = idx_end
		
		lb, ub = self.idx_range[0], self.idx_range[1]
		batch_size = self.batch_size
		n_batches = (ub - lb) // batch_size + 1
		indices = [i for i in range(lb, ub + 1)]
		return indices		
	

	def permute(self):
		"""
		Shuffle indices after each epoch
		"""
		self.indices = np.random.permutation(self.indices)
		

	def __iter__(self):
		"""
		Iterator itself
		"""
		return (self.indices[i] for i in range(len(self.indices)))
				

	def __len__(self):
		"""
		Get the length of all the valid indices
		"""
		return len(self.indices)
		
