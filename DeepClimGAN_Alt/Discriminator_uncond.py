import numpy as np
import torch
from ops import *
from Constants import clmt_vars, n_channels_in_diff_vct
import logging

"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""

n_channels = len(clmt_vars)
class Discriminator(nn.Module):
	def __init__(self, label_smoothing, is_autoencoder, z_shape):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			#model takes n_channels+2 as number of channels,
			#since we are doing unconditional GAN
			#n_channels is the samew if we are doing unconditional model or taking just highres context
                        conv3d(n_channels, 128),
			lrelu(0.2),
			conv3d(128, 256),
			batchNorm5d(256, 1e-3),
			lrelu(0.2),
			conv3d(256, 512),
			batchNorm5d(512, 1e-3),
			lrelu(0.2),
			conv3d(512,2))
		#self.autoencoder = is_autoencoder
		self.fc1 = torch.nn.Linear(2 * 8 * 16 * 2, 128)
		self.fc2 = torch.nn.Linear(128, 1)
		self.fc3 = torch.nn.Linear(2 * 8 * 16 * 2, 256)
		self.fc4 = torch.nn.Linear(256, 128)
		self.encoder = torch.nn.Linear(128, z_shape)
		self.sigmoid = torch.nn.Sigmoid()		


	def forward(self, x):
		out = self.model(x)
		b, h, w, t, ch = out.shape
		out = out.view(b, h * w * t * ch)				
		#res = out
		out_256 = self.fc3(out)
		out_128 = self.fc4(out_256)
		out = self.fc2(out_128)
		#out_128 = self.fc1(out)
		#out = self.fc2(out_128)
		#res = out_256
		#lin1 = self.fc1(out)
		#lin2 = self.fc2(lin1)
		res = self.sigmoid(out)
		#res = self.encoder(lin1)
		#res = self.sigmoid(out)
		#res = out
		#res = out_256
		return res
