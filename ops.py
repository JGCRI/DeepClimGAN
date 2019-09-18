import torch.nn as nn

def conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
	return nn.Conv3d(in_channels, out_channels, kernel_size =  kernel_size, stride = stride, padding = padding, bias = True)

def upconv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
	return nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def upconv3d_same(in_channels, out_channels, kernel_size=3, stride = 1, padding = 1):
	return nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)


def batchNorm5d(num_features, eps = 1e-5):
	return nn.BatchNorm3d(num_features, eps = eps)

#input size is 2 x V x 128 x 256 x 32 or V x 128 x 256 x 32
def layerNorm(features):
	return nn.LayerNorm(features)

def relu(inplace = True):
	return nn.ReLU(inplace)

def lrelu(negative_slope = 0.2, inplace = True):
	return nn.LeakyReLU(negative_slope, inplace)

#def conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
#	return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

#def pool2d(kernel_size=4, stride=2, padding=1):
#	return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

def pool2d(kernel_size=4, stride=2, padding=1):
	return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

def fc(D_in, D_out):
	return nn.Linear(D_in, D_out)
