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
	def __init__(self, h, w, t, ch, batch_size):
		super(Generator, self).__init__()
		
		self.pool = pool2d()
		self.relu = relu()
		self.relu2 = nn.ReLU(inplace=True)
		self.sigm = torch.nn.Sigmoid()
		init_ctxt_channel_size = self.get_init_channel_size(t, ch)
			
		#block 0
		self.fc1 = fc(100, 512)
		self.fc2 = fc(512, 4096)
		
		#block 1
		self.upconv1 = upconv3d(128, 512)
		self.batchNorm5d_1 = batchNorm5d(512+64+32)
		
		#block 2
		self.upconv2 = upconv3d(512+64+32, 256)
		self.batchNorm5d_2 = batchNorm5d(256+16+32)
		
		#block 3
		self.upconv3 = upconv3d(256+16+32, 128)
		self.batchNorm5d_3 = batchNorm5d(128+16+8)

		#block 4
		self.upconv4 = upconv3d(128+16+8, 64)
		self.batchNorm5d_4 = batchNorm5d(64+8+4)
		
		#block 5
		self.upconv5 = upconv3d(64+8+4, 32)
		self.batchNorm5d_5 = batchNorm5d(32+2+init_ctxt_channel_size)
		
		#block 6
		self.upconv6 = upconv3d_same(init_ctxt_channel_size+2+32, n_channels)

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
		x = self.fc2(self.fc1(x)).reshape(batch_size, 128, 4, 8, 1)
				
		#block 1
		x = self.upconv1(x)
		rep_factor = x.shape[-1] #time dimension of x, we should make contexts to match this size
		init1 = self.pool(self.conv32_64(self.pool(self.conv16_32(self.pool(self.conv8_16(self.pool(self.init1(high_res_context))))))))
		init1 = init1.unsqueeze(-1).repeat(1, 1, 1, 1, rep_factor)
		avg1 = self.pool(self.conv16_32(self.pool(self.conv8_16(self.pool(self.conv4_8(self.pool(self.conv2_4(avg_context))))))))
		avg1 = avg1.unsqueeze(-1).repeat(1, 1, 1, 1, rep_factor)		
		x = torch.cat([x, avg1, init1], dim=1)#concat across feature channels
		x = self.relu(self.batchNorm5d_1(x))
		
		#block 2
		x = self.upconv2(x)
		rep_factor = x.shape[-1]
		init2 = self.pool(self.conv16_32(self.pool(self.conv8_16(self.pool(self.init1(high_res_context))))))
		init2 = init2.unsqueeze(-1).repeat(1,1,1,1, rep_factor)
		avg2 = self.pool(self.conv8_16(self.pool(self.conv4_8(self.pool(self.conv2_4(avg_context))))))
		avg2 = avg2.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, avg2, init2],dim=1)
		x = self.relu(self.batchNorm5d_2(x))
		
		#block3
		x = self.upconv3(x)
		rep_factor = x.shape[-1]
		init3 = self.pool(self.conv8_16(self.pool(self.init1(high_res_context)))) 
		init3 = init3.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		avg3 = self.pool(self.conv4_8(self.pool(self.conv2_4(avg_context))))
		avg3 = avg3.unsqueeze(-1).repeat(1,1,1,1,rep_factor)		
		x = torch.cat([x, avg3, init3],dim=1)
		x = self.relu(self.batchNorm5d_3(x))

		#block4
		x = self.upconv4(x)
		rep_factor = x.shape[-1]
		init4 = self.pool(self.init1(high_res_context)) 
		init4 = init4.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		avg4 = self.pool(self.conv2_4(avg_context))
		avg4 = avg4.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, avg4, init4],dim=1)
		x = self.relu(self.batchNorm5d_4(x))
			
		#block5
		x = self.upconv5(x)
		rep_factor = x.shape[-1]
		init5 = high_res_context.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		avg5 = avg_context.unsqueeze(-1).repeat(1,1,1,1,rep_factor)
		x = torch.cat([x, init5, avg5],dim=1)
		x = self.relu(self.batchNorm5d_5(x))
		
		#block 6
		out = self.upconv6(x)
		
		#apply activation functions per channels: pr -> relu, rhs, rhsmin, rhsmax -> sigmoid
		for i, (key, val) in enumerate(clmt_vars.items()):
			if i == 0:
				out[:,i] = self.relu2(out[:,i].clone())
			elif i == 4 or i == 5 or i == 6:
				out[:,i] = self.sigm(out[:,i].clone())

		
		return out

