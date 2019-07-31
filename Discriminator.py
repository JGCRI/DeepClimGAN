import numpy as np
import torch
from ops import *
from Constants import clmt_vars


"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""
n_channels = len(clmt_vars)

class Discriminator(nn.Module):
	def __init__(self, label_smoothing):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			#model takes n_channels+2 as number of channels,
			#since we are doing conditional GAN (2 channels is for average maps)
			conv3d(n_channels+2, 128),
			lrelu(0.2),
			conv3d(128, 256),
			batchNorm5d(256, 1e-3),
			lrelu(0.2),
			conv3d(256, 512),
			batchNorm5d(512, 1e-3),
			lrelu(0.2),
			conv3d(512,2))
		
		#todo: double check
		self.fc1 = torch.nn.Linear(4 * 8 * 2, 16)
		self.fc2 = torch.nn.Linear(16, 1)
		
	def forward(self, x):
		out = self.model(x)
		output = self.fc2(self.fc1(out))
		return output
