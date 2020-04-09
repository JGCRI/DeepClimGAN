import numpy as np
import torch
import torch.nn as nn

from ops import *
from Constants import clmt_vars
import logging


"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""

n_channels = len(clmt_vars.items())

class Generator(nn.Module):
	def __init__(self, h, w, t, ch, batch_size, z_shape,  last_layer_size):
		super(Generator, self).__init__()
		
		self.pool = pool2d()
		self.relu = relu()
		self.relu2 = nn.ReLU(inplace=True)
		self.sigm = torch.nn.Sigmoid()
		self.last_layer_size = last_layer_size
		
		#block 0
		self.fc1 = fc(z_shape, 512)
		self.fc2 = fc(512, 4096)
		
		#block 1
		self.upconv1 = upconv3d(128, 512)
		n_features = 512
		self.batchNorm5d_1 = batchNorm5d(n_features)
		
		#block 2
		self.upconv2 = upconv3d(512, 256)
		n_features = 256
		self.batchNorm5d_2 = batchNorm5d(n_features)
		
		#block 3
		self.upconv3 = upconv3d(256, 128)
		n_features = 128
		self.batchNorm5d_3 = batchNorm5d(n_features)

		#block 4
		self.upconv4 = upconv3d(128, 64)
		n_features = 64
		self.batchNorm5d_4 = batchNorm5d(n_features)
		
		#block 5
		self.upconv5 = upconv3d(64, 32)
		n_features = 32
		self.batchNorm5d_5 = batchNorm5d(32)
		
		#block 6
		self.batchNorm5d_6 = batchNorm5d(last_layer_size)
		self.upconv6 = conv3d_same(32, last_layer_size)
		self.upconv6_for_smoothing = conv3d_same(last_layer_size, last_layer_size)
		self.upconv6_final = conv3d_same(last_layer_size, n_channels)


		#convolutions for the context
		#self.init1 = conv2d(init_ctxt_channel_size, 8)				
		self.conv2_4 = conv2d(2, 4)
		self.conv4_8 = conv2d(4, 8)
		self.conv8_16 = conv2d(8, 16)
		self.conv16_32 = conv2d(16, 32)
		self.conv32_64 = conv2d(32, 64)



	def get_init_channel_size(self, t, ch):
		return t * ch


	def forward(self, x, avg_context, high_res_context):
		"""
		param: x (tensor) B x C x H x W x T, B x 2 x H x W, B x C x H x W x T
		param: avg_context (tensor)
		param: high_res_context (tensor)
		"""

		#modificaiton if using only precipitatin in a context -> copy across features dimension
		#avg_context = torch.cat((avg_context, avg_context), 1)

		#block 0
		batch_size = x.shape[0]
		fc1 = self.fc1(x)
		fc2 = self.fc2(fc1)

		x = fc2.reshape(batch_size, 128, 4, 8, 1)
				
		#block 1
		x = self.upconv1(x)
		x = self.relu(self.batchNorm5d_1(x))
		
		
		#block 2
		x = self.upconv2(x)
		x = self.relu(self.batchNorm5d_2(x))
		
		#block3
		x = self.upconv3(x)
		x = self.relu(self.batchNorm5d_3(x))


		#block4
		x = self.upconv4(x)
		x = self.relu(self.batchNorm5d_4(x))

	
		#block5
		x = self.upconv5(x)
		x = self.relu(self.batchNorm5d_5(x))
		
		#block 6
		x = self.upconv6(x)
		out = x

		keys = list(clmt_vars.keys())
		pr_idx = keys.index('pr') if 'pr' in keys else None		
		tas_idx = keys.index('tas') if 'tas' in keys else None
		tasmin_idx = keys.index('tasmin') if 'tasmin' in keys else None
		tasmax_idx = keys.index('tasmax') if 'tasmax' in keys else None
		rhs_idx = keys.index('rhs') if 'rhs' in keys else None
		rhsmin_idx = keys.index('rhsmin') if 'rhsmin' in keys else None
		rhsmax_idx = keys.index('rhsmax') if 'rhsmax' in keys else None

		#logging.info("tensor size : {}".format(out.shape))
		if pr_idx:
			#apply activation functions per channels: pr -> relu, rhsmin -> sigmoid
			out[:,pr_idx] = self.relu2(out[:,pr_idx].clone())
			#out[:,rhsmin_idx] = self.sigm(out[:,rhsmin_idx].clone())

		"""
		#apply reparametrization
		if tas_idx:
			out[:,tas_idx] = out[:,tasmin_idx] + torch.exp(out[:,tas_idx].clone())
			out[:,tasmax_idx] = out[:,tas_idx] + torch.exp(out[:,tasmax_idx].clone())
		
		if rhs_idx:
			#first update daily avg rhs
			out[:,rhs_idx] = self.sigm(out[:,rhsmin_idx] + torch.exp(out[:,rhs_idx].clone()))
		
			#then use it to update max rhs
			out[:,rhsmax_idx] = self.sigm(out[:,rhs_idx] + torch.exp(out[:,rhsmax_idx].clone()))

		"""
		return out

