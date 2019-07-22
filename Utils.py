import sys
from torch.autograd import Variable
import csv
import os
import numpy as np
import torch.nn as nn
import torch

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

def to_variable(x, device,requires_grad=True):
	x = x.to(device)
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
	

class GaussianNoise(nn.Module):
	def __init__(self, device, stddev=0.2, is_relative_detach=True, is_training=True):
		super().__init__()
		self.stddev= stddev
		self.is_relative_detach = is_relative_detach
		self.is_training = is_training
		self.device = device

	def forward(self, x):
		if self.is_training and self.stddev != 0:
			noise = Variable(torch.randn(x.size()).to(self.device) * self.stddev)
			x = x + noise
		return x
			
		
