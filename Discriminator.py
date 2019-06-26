import numpy as np
import torch
from ops import *
from Constants import clmt_vars



"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""
n_channels = len(clmt_vars)

class Discriminator:
	def __init__(self):
		self.model = nn.Sequential(
			conv3d(n_channels, 128),
			lrelu(0.2),
			conv3d(128, 256),
			batchNorm5d(256, 1e-3),
			lrelu(0.2),
			conv3d(256, 512),
			batchNorm5d(512, 1e-3),
			lrelu(0.2),
			conv3d(512,1))

	def forward(self, x):
		out = self.model(x).squeeze()
		return out

