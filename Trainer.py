import numpy as np
import os
import sys
from NETCDFDataset import NETCDFDataset
from Discriminator import Discriminator
from Generator import Generator
from Normalizer import Normalizer
import torch
from torch.utils import data
from Constants import clmt_vars
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def weights_init(m):
	"""
	Custom weights initialization called on Generator and Discriminator
	All weights are initialized from a zero-centered Normal Distribution
	with std=0.02
	 param: m ()
	return: None
	"""
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
	        m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
	        m.weight.data.normal_(1.0, 0.02)
	        m.bias.data.fill_(0)


def to_variable(x, requires_grad = True):
    x = x.to(device)
    return Variable(x, requires_grad)


#Default params
data_dir = '../clmt_data/'
#Percent of data to use for training
train_pct = 0.7
#Percent of data to use at all
data_pct = 1

#dimension of Gaussian distribution to sample noise from
z_shape = 100
#number of iterations to update Discrimnator (default = 1)
k = 1
#normalize data
apply_norm = True

num_epoch = 5
batch_size = 1
lr = 0.0002 #NOTE: this lr is coming from original paper about DCGAN
l1_lambda = 10
lon, lat, t, channels = 128, 256, 30, len(clmt_vars)
context_length = 5


netD = Discriminator()
netG = Generator(lon, lat, context_length, channels, batch_size)
netD.apply(weights_init)
netG.apply(weights_init)


#use all possible GPU cores available
#if torch.cuda.device_count() > 1:
#    print("Using ", torch.cuda.device_count(), "GPUs!")
#    netD = nn.DataParallel(netD)
#    netG = nn.DataParallel(netG)
netD.to(device)
netG.to(device)


loss_func = torch.nn.CrossEntropyLoss()
d_optim = torch.optim.Adam(netD.parameters(), lr, [0.5, 0.999])
g_optim = torch.optim.Adam(netG.parameters(), lr, [0.5, 0.999])


print("Started parsing data...")
ds = NETCDFDataset(data_dir, data_pct,train_pct)

if apply_norm:
	normalizer = Normalizer()
	#normalize training set
	ds.normalized_train = normalizer.normalize(ds, 'train')


#Specify that we are loading training set
dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)



for current_epoch in range(1, num_epoch+1):
	for batch_idx, batch in enumerate(dl):
		current_month, avg_context, high_res_context = batch
		real_labels = to_variable(torch.LongTensor(np.ones(batch_size, dtype = int)), requires_grad = False)
		fake_labels = to_variable(torch.LongTensor(np.zeros(batch_size, dtype = int)), requires_grad = False)
		input = ds.build_input_for_D(current_month, avg_context, high_res_context)
		#move batch to GPU
		input = input.to(device)

		for i in range(k):
			netD.zero_grad()
			netG.zero_grad()
			outputs = netD(input)
			bsz, ch, h, w, t = outputs.shape
			outputs = outputs.view(bsz, h * w * t)
			d_real_loss = loss_func(outputs, real_labels)
			z = ds.get_noise(z_shape, batch_size)
			high_res_context = high_res_context.reshape(bsz, context_length*channels, lon, lat)
			avg_context = torch.mean(avg_context, -1)
			fake_inputs = netG(z, avg_context, high_res_context)
			outputs = netD(fake_inputs).view(batch_size, h * w * ch)
			d_fake_loss = loss_func(outputs, fake_labels)
			d_loss = -(d_real_loss + d_fake_loss)
			d_loss.backward()
			d_optim.step()
		print("epoch {}, step {},  d loss = {}".format(current_epoch, k,d_loss.item()))

		#train Generator
		netD.zero_grad()
		netG.zero_grad()
		noise = ds.get_noise(z_shape, batch_size)
		g_outputs_fake = netG(noise, avg_context, high_res_context)
		d_input = ds.build_input_for_D(g_outputs_fake, avg_context, high_res_context)
		outputs = netD(d_input)
		bsz, c, h, w, t = outputs.shape
		outputs = outputs.view(bsz, h * w * t)
		g_loss = (-1) * loss_func(outputs, fake_labels)
		g_loss.backward()
		g_optim.step()
		
		print("epoch {}, g_loss = {}\n".format(current_epoch,g_loss.item()))

