import torch.nn as nn



def conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
	return nn.Conv3d(in_channels, out_channels, kernel_size =  kernel_size, stride = stride, padding = padding, bias = True)

def deconv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
	return nn.ConTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def batchNorm5d(num_features, eps = 1e-5):
	return nn.BatchNorm3d(num_features, eps = eps)

def relu(inplace = True):
	return nn.ReLU(inplace)

def lrelu(negative_slope = 0.2, inplace = True):
	return nn.LeakyReLU 
