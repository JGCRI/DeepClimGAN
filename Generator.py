import numpy as np
import torch
import torch.nn as nn

from ops import *
from Constants import clmt_vars

"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
"""
n_channels = len(clmt_vars)

# w'  = (w - 2p +  k )/ s + 1
# 5s = 128 + k, k = 5s - 128
# 9s = 256 + k = 256 + 5s - 128
# 4s = 128, s = 32, k = 5 * 32 - 128
#
# 9s = 128 + k, k = 9s - 128
# 17s = 256 + k = 256 - 128 + 9s
# 8s = 128, s = 16, k = 16

class Generator(nn.Module):
	def __init__(self, h, w, t, ch, batch_size):
		super(Generator, self).__init__()
		self.pool = pool2d()

		#block 1
		self.fc1 = fc(100, 512)
		self.fc2 = fc(512, 2048)
		self.upconv1 = upconv3d(256, 512,2, 2, 0)
		init_ctxt_channel_size = self.get_init_channel_size(t, ch)
		self.init1 = conv2d(init_ctxt_channel_size, 512, 16, 16, 0)
		#number 2 is two average map (precipitation and temperature)
		self.avg1 = conv2d(2, 512,16, 16, 0)
		self.batchNorm5d_1 = batchNorm5d(3*512)
		self.relu = relu()
		
		#block 2
		self.upconv2 = upconv3d(3*512, 256, 2, 2, 0)
		self.init2 = conv2d(init_ctxt_channel_size, 256, 8, 8, 0)
		self.avg2 = conv2d(2, 256, 8, 8, 0)
		self.batchNorm5d_2 = batchNorm5d(3*256)
		
		#block 3
		self.upconv3 = upconv3d(3*256, 128, 2, 2, 0)
		self.init3 = conv2d(init_ctxt_channel_size, 128, 4, 4, 0)
		self.avg3 = conv2d(2, 128, 4, 4, 0)
		self.batchNorm5d_3 = batchNorm5d(3*128)

		#block 4
		self.upconv4 = upconv3d(3*128, 64, 2, 2, 0)
		self.init4 = conv2d(init_ctxt_channel_size, 64, 2, 2, 0)
		self.avg4 = conv2d(2, 64, 2, 2, 0)
		self.batchNorm5d_4 = batchNorm5d(3*64)
		
		#block 5
		self.upconv5 = upconv3d(3*64, n_channels)
		self.tanh = nn.Tanh()



	def get_init_channel_size(self, t, ch):
		return t * ch


	def forward(self, x, avg_context, high_res_context):
		"""
		param: x (tensor) B x C x H x W x T, B x 2 x H x W, B x C x H x W x T
		param: avg_context (tensor)
		param: high_res_context (tensor)
		"""
		
		#block 1
		batch_size = x.shape[0]
		x = self.fc2(self.fc1(x)).reshape(batch_size, 256, 2, 4, 1)
		x = self.upconv1(x)
		rep_factor = x.shape[-1] #time dimension of x, we should make contexts to match this size
		init = self.pool(self.init1(high_res_context)).unsqueeze(-1).repeat(1, 1, 1, 1, rep_factor)
		avg = self.pool(self.avg1(avg_context)).unsqueeze(-1).repeat(1, 1, 1, 1, rep_factor)
		x = torch.cat([x, avg, init], dim=1)#concat across feature channels
		x = self.relu(self.batchNorm5d_1(x))
		
		#block 2
		x = self.upconv2(x)
		rep_factor = x.shape[-1]
		avg = self.pool(self.avg2(avg_context)).unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		init = self.pool(self.init2(high_res_context)).unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		print(x.shape, avg.shape, init.shape)
		x = torch.cat([x, avg, init],dim=1)
		x = self.relu(self.batchNorm5d_2(x))

		#block3
		x = self.upconv3(x)
		rep_factor = x.shape[-1]
		avg = self.pool(self.avg3(avg_context)).unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		init = self.pool(self.init3(high_res_context)).unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, avg, init],dim=1)
		x = self.relu(self.batchNorm5d_3(x))
		
		#block4
		x = self.upconv4(x)
		rep_factor = x.shape[-1]
		avg = self.pool(self.avg4(avg_context)).unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		init = self.pool(self.init4(high_res_context)).unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, avg, init],dim=1)
		x = self.relu(self.batchNorm5d_4(x))
		
		#block5
		x = self.upconv5(x)
		out = self.tanh(x)

		return out

