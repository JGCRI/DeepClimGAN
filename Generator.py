import numpy as np
import torch

from layers import *
from Constants import clmt_vars

"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""
n_channels = len(clmt_vars)

class Generator(nn.Module):
	def __init__(self):
		self.model = nn.Sequential(
			deconv3d(512,512),
			batchNorm5d(512),
			relu(),
			deconv3d(512,256),
			batchNorm5d(256),
			relu(),
			deconv3d(256,128),
			batchNorm5d(128),
			relu(),
			deconv3d(128,64),
			batchNorm5d(64),
			relu())

		self.out_net = nn.Sequential(deconv3d(64, n_channels), nn.Tanh())
	
	
	def forward(self, x):
		x = self.model(x)
		out = self.out_net(x)
		return out



