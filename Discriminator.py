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
		self.label_smoothing = label_smoothing
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
		#self.sig = nn.Sequential(nn.Linear(2,1), nn.Sigmoid())

	def forward(self, x):
		out = self.model(x)
		#if smoothing, apply sigmoid for KLDivLoss
		#if self.label_smoothing:
			#print(out.shape)	
			#TODO:
			#ch, h, w, t = out.shape
			#out = out.view(-1, ch * h * w *t)
			#out = self.sig(out)
			#out = torch.sigmoid(out)
			#print(out.shape)
		return out
