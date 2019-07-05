import numpy as np
import torch
import torch.nn as nn

from ops import *
from Constants import clmt_vars

"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""
n_channels = len(clmt_vars)

class Generator(nn.Module):
	def __init__(self, h, w, t, ch):
		super(Generator, self).__init__()
		self.pool = pool2d()
		self.upconv1 = upconv3d(512, 512)
		init_ctxt_channel_size = self.get_init_channel_size(t, ch)
		self.init11 = conv2d(init_ctxt_channel_size, 512)
		
		#number 2 is two average mapd (precipitation and temperature)
		self.avg1 = conv2d(2, 512)
		self.batchNorm5d_1 = batchNorm5d(512)
		self.relu = relu()
		
		self.upconv2 = upconv3d(512, 256)
		self.init2 = conv2d(init_ctxt_channel_size, 256)
		self.avg2 = conv2d(2, 256)
		self.batchNorm5d_2 = batchNorm5d(256)
		
		
		self.upconv3 = upconv3d(256, 128)
		self.init3 = conv2d(init_ctxt_channel_size, 128)
		self.avg3 = conv2d(2, 64)
		self.batchNorm5d_3 = batchNorm5d(128)
		
		self.upconv4 = upconv3d(128, 64)
		self.init4 = conv2d(init_ctxt_channel_size, 64)
		self.avg4 = conv2d(init_ctxt_channel_size, 64)
		self.batchNorm5d_4 = batchNorm5d(64)

		self.upconv5 = upconv3d(64, n_channels)
		self.tanh = nn.Tanh()


		
	def get_init_channel_size(self, t, ch):
		return t * ch


	def forward(self, x, avg_context, high_res_context):
		#reshape high_res_context to have the size HxWx1xKV
		h, w, t, ch = high_res_ctxt.shape
		high_res_ctxt = high_res_ctxt.resize_(h, w, 1, t*ch)
		
		
		#block 1
		x = self.upconv1(x)
		avg = self.avg1(avg_context)
		init = self.init1(high_res_context)
		x = torch.cat([x, avg, init], dim=2)#concat across time
		x = relu(batchNorm5d_1(x))

		#block 2
		x = self.upconv2(x)
		avg = self.avg2(avg_context)
		init = self.init2(high_res_context)
		x = torch.cat([x, avg, init],dim=2)
		x = relu(batchNorm5d_2(x))

		#block3
		x = upconv3(x)
		avg = self.avg3(avg_context)
		init = self.init3(high_res_context)
		x = torch.cat([x, avg, init],dim=2)
		x = relu(batchNorm5d_3(x))
		
		#block4
		x = upconv4(x)
		avg = self.avg4(avg_context)
		init = self.init4(high_res_context)
		x = torch.cat([x, avg, init],dim=2)
		x = relu(batchNorm5d_4(x))
		
		#block5
		x = upconv5(x)
		out = tanh(x)

		return out



