import numpy as np
import torch
import torch.nn as nn

from ops import *
from Constants import clmt_vars
import logging


"""
reference: https://github.com/batsa003/videogan/blob/master/model.py
Alterantive Generator, which means instead of doing transposed convolutions, 
we are implementing upsample + same convolutional operation,
that helps to smooth out the grid on the map.
"""
n_channels = len(clmt_vars.items())

class Generator(nn.Module):
	def __init__(self, h, w, t, ch, batch_size, z_shape,  last_layer_size):
		super(Generator, self).__init__()
		self.pool = pool2d()
		self.relu = relu()
		self.relu2 = nn.ReLU(inplace=True)
		self.sigm = torch.nn.Sigmoid()
		init_ctxt_channel_size = self.get_init_channel_size(t, ch)
		self.last_layer_size = last_layer_size
		self.upsample = upsample_layer(2)#scale factor is 2
		
		#block 0
		self.fc1 = fc(z_shape, 512)
		self.fc2 = fc(512, 4096)
		
		#block 1
		self.conv1 = conv3d_same(128, 512)
		n_features = 512 + 64 + 32
		self.batchNorm5d_1 = batchNorm5d(n_features)
		
		#block 2
		self.conv2 = conv3d_same(512+64+32, 256)
		n_features = 256 + 32 + 16
		self.batchNorm5d_2 = batchNorm5d(n_features)
		
		#block 3
		self.conv3 = conv3d_same(256+16+32, 128)
		n_features = 128 + 16 + 8
		self.batchNorm5d_3 = batchNorm5d(n_features)

		#block 4
		self.conv4 = conv3d_same(128+16+8, 64)
		n_features = 64 + 8 + 4
		self.batchNorm5d_4 = batchNorm5d(n_features)
		
		#block 5
		self.conv5 = conv3d_same(64+8+4, 32)
		n_features = 32 + 2 + init_ctxt_channel_size
		self.batchNorm5d_5 = batchNorm5d(32+2+init_ctxt_channel_size)
		
		#block 6
		self.batchNorm5d_6 = batchNorm5d(last_layer_size)
		self.upconv6 = conv3d_same(init_ctxt_channel_size+2+32, last_layer_size)
		self.upconv6_for_smoothing = conv3d_same(last_layer_size, last_layer_size)
		self.upconv6_final = conv3d_same(last_layer_size, n_channels)

		#convolutions for the context
		self.init1 = conv2d(init_ctxt_channel_size, 8)				
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

		#block 0
		batch_size = x.shape[0]
		fc1 = self.fc1(x)
		fc2 = self.fc2(fc1)

		#fc2 = self.fc2(x)#for pretraijing for z = 512
		x = fc2.reshape(batch_size, 128, 4, 8, 1)
				
		#block 1
		x = self.conv1(self.upsample(x))
		rep_factor = x.shape[-1] #time dimension of x, we should make contexts to match this size
		init1 = self.pool(self.conv32_64(self.pool(self.conv16_32(self.pool(self.conv8_16(self.pool(self.init1(high_res_context))))))))
		init1 = init1.unsqueeze(-1).repeat(1, 1, 1, 1, rep_factor)
		avg1 = self.pool(self.conv16_32(self.pool(self.conv8_16(self.pool(self.conv4_8(self.pool(self.conv2_4(avg_context))))))))
		avg1 = avg1.unsqueeze(-1).repeat(1, 1, 1, 1, rep_factor)		
		x = torch.cat([x, avg1, init1], dim=1)#concat across feature channels
		x = self.relu(self.batchNorm5d_1(x))
		
		
		#block 2
		x = self.conv2(self.upsample(x))
		rep_factor = x.shape[-1]
		init2 = self.pool(self.conv16_32(self.pool(self.conv8_16(self.pool(self.init1(high_res_context))))))
		init2 = init2.unsqueeze(-1).repeat(1,1,1,1, rep_factor)
		avg2 = self.pool(self.conv8_16(self.pool(self.conv4_8(self.pool(self.conv2_4(avg_context))))))
		avg2 = avg2.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, avg2, init2],dim=1)
		x = self.relu(self.batchNorm5d_2(x))
		
		#block3
		x = self.conv3(self.upsample(x))
		rep_factor = x.shape[-1]
		init3 = self.pool(self.conv8_16(self.pool(self.init1(high_res_context)))) 
		init3 = init3.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		avg3 = self.pool(self.conv4_8(self.pool(self.conv2_4(avg_context))))
		avg3 = avg3.unsqueeze(-1).repeat(1,1,1,1,rep_factor)		
		x = torch.cat([x, avg3, init3],dim=1)
		x = self.relu(self.batchNorm5d_3(x))


		#block4
		x = self.conv4(self.upsample(x))
		rep_factor = x.shape[-1]
		init4 = self.pool(self.init1(high_res_context)) 
		init4 = init4.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		avg4 = self.pool(self.conv2_4(avg_context))
		avg4 = avg4.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, avg4, init4],dim=1)
		x = self.relu(self.batchNorm5d_4(x))

	
		#block5
		x = self.conv5(self.upsample(x))
		rep_factor = x.shape[-1]
		init5 = high_res_context.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		avg5 = avg_context.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, init5, avg5],dim=1)
		x = self.relu(self.batchNorm5d_5(x))
		
		#block 6
		x = self.upconv6(x)
		out = x

		keys = list(clmt_vars.keys())
		pr_idx = keys.index('pr')		
		tas_idx = keys.index('tas')
		tasmin_idx = keys.index('tasmin')
		tasmax_idx = keys.index('tasmax')
		rhs_idx = keys.index('rhs')
		rhsmin_idx = keys.index('rhsmin')
		rhsmax_idx = keys.index('rhsmax')

		#apply activation functions per channels: pr -> relu, rhsmin -> sigmoid
		out[:,pr_idx] = self.relu2(out[:,pr_idx].clone())
		out[:,rhsmin_idx] = self.sigm(out[:,rhsmin_idx].clone()) #MAYBE LATER

		#apply reparametrization
		out[:,tas_idx] = out[:,tasmin_idx] + torch.exp(out[:,tas_idx].clone())
		out[:,tasmax_idx] = out[:,tas_idx] + torch.exp(out[:,tasmax_idx].clone())
		
		#first update daily avg rhs
		out[:,rhs_idx] = self.sigm(out[:,rhsmin_idx] + torch.exp(out[:,rhs_idx].clone())) #MAYBE LATER
		
		#then use it to update max rhs
		out[:,rhsmax_idx] = self.sigm(out[:,rhs_idx] + torch.exp(out[:,rhsmax_idx].clone())) #MAYBE LATER

		return out

