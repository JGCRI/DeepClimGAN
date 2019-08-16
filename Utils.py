import sys
from torch.autograd import Variable
import csv
import os
import numpy as np
import torch.nn as nn
import torch
from Constants import scenarios


def weights_init(m):
	"""
	Custom weights initializaton called on G and D
	All weights are iitializaed from a zero-centerd Normal Distribution
	with std=0.01
	param: m
	
	"""

	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def to_variable(x, requires_grad=True):
	return Variable(x, requires_grad)

def save_grads(model, model_name):
	'''
	Intended to be used to check the gradients.
	Use before zeroing out gradients.
	usage:
		save_gradients(netG, "Generator")
		netG.zero_grad()
	'''

	grads = []
	for p in model.parameters():
		if not p.grad is None:
			grads.append(float(p.grad.mean()))
	grads = np.array(grads)
	if len(grads) > 0:
		return np.linalg.norm(grads)
	elif len(grads) == 0:
		return 0



#save losses and grads
def save_results(sys, losses, grads):

	#exp_id = str(sys.argv[1])
	#to_save_fd = sys.argv[4]
	#exp_id_dir = "{}exp{}/".format(to_save_fd, exp_id)
	exp_id_dir = sys.argv[4]
	
	D_losses, D_real_losses, D_fake_losses, G_losses = losses
	D_grads, G_grads = grads

		
	with open(exp_id_dir + 'g_losses.csv', 'w') as file:
		for l in G_losses:
			file.write(str(l))
			file.write('\n')

	with open(exp_id_dir + 'd_losses.csv', 'w') as file:
		for l in D_losses:
			file.write(str(l))
			file.write('\n')
	with open(exp_id_dir + 'd_real_losses.csv', 'w') as file:
		for l in D_real_losses:
			file.write(str(l))
			file.write('\n')
	
	with open(exp_id_dir + 'd_fake_losses.csv', 'w') as file:
		for l in D_fake_losses:
			file.write(str(l))
			file.write('\n')

	with open(exp_id_dir + 'g_grads.csv', 'w') as file:
		for g in G_grads:
			file.write(str(g))
			file.write('\n')

	with open(exp_id_dir + 'd_grads.csv', 'w') as file:
		for d in D_grads:
			file.write(str(d))
			file.write('\n')
	

def sort_files_by_size(directory):
	# Get all files.
	list = os.listdir(directory)

	# Loop and add files to list.
	pairs = []
	for file in list:
    		# Use join to get full file path.
    		location = os.path.join(directory, file)

    		# Get size and add to list of tuples.
    		size = os.path.getsize(location)
    		pairs.append((size, file))

	# Sort list of tuples by the first element, size.
	pairs.sort(key=lambda s: s[0], reverse=True)
	return pairs

def partition_data_between_nodes(sorted_files, node_size, empty_space, n_proc_per_node):
	partition = {}
	node_rank = 0
	size = 0
	node_size_per_proc = (node_size - empty_space) // n_proc_per_node
	for i, (f_size, file) in enumerate(sorted_files):
		size += f_size
		if size < node_size_per_proc and size != f_size:
			partition[node_rank].append(file)
		else:
			if size > node_size_per_proc:
				node_rank += 1
			size = f_size
			partition[node_rank] = [file]
	return partition



def snake_data_partition(sorted_files, world_size):
	partition = {}
	n_processes = world_size
	file_groups = []
	i = 0

	N = len(sorted_files)
	
	files = []
	for file in sorted_files:
		files.append(file[1])
	sorted_files = files
	
	#split files per groups
	while i < N:
		if i + n_processes > N:
			group = sorted_files[i:]
		else:
			group = sorted_files[i:i+n_processes]
		
		file_groups.append(group)
		
		i += n_processes	

	#distribute files between processes
	i = 0
	N_groups = len(file_groups)
	
	while i < N_groups:
		group = file_groups[i]
		group_len = len(group)
		#first group goes from left to right
		#then from right to left
		if i % 2 == 0:
			if i == 0:
				for j in range(0,group_len):
					partition[j] = [group[j]]
			else:
				for j in range(0,group_len):
					partition[j].append(group[j])
		else:
			group = group[::-1]
			for j in range(group_len):
				part = partition[world_size - 1 - j]
				part.append(group[j])

		
		i += 1
	return partition


def get_node_size(file_name):
        GB_to_KB = 1048576
        with open(file_name, 'r') as of:
                line = of.readlines(0)[0]
                info = line.split(" ")
                size = int(info[-2]) * GB_to_KB
                return size



class GaussianNoise(nn.Module):
	def __init__(self, device, stddev=0.2, is_relative_detach=True, is_training=True):
		super().__init__()
		self.stddev= stddev
		self.is_relative_detach = is_relative_detach
		self.is_training = is_training
		self.device = device

	def forward(self, x):
		if self.is_training and self.stddev != 0:
			tsr = torch.randn(x.size()).cuda()
			noise = Variable(tsr * self.stddev)
			x = x + noise
		return x
			
		
