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

def weights_init(m):
        """
        Custom weights initialization called on	Generator and Discriminator
       	All weights are initialized from a zero-centered Normal Distribution
	with std=0.02
	 param: m ()

        return:	None
        """
	
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

def to_gpu(x):
	if torch.cuda.is_available():
		#TODO: check of tensors are transfered succesfully to GPU
		x = x.cuda()

def to_variable(x, requires_grad = True):
	to_gpu(x)
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
batch_size = 64
lr = 0.0002 #NOTE: this lr is coming from original paper about DCGAN
l1_lambda = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


h, w, t, c = 128, 256, 30, len(clmt_vars)


netD = Discriminator()
netG = Generator(h, w, t, c)
netD.apply(weights_init)
netG.apply(weights_init)

#use all possible GPU cores available
if torch.cuda.device_count() > 1:
	print("Using ", torch.cuda.device_count(), "GPUs!")
	netD = nn.DataParallel(netD)
	netG = nn.DataParallel(netG)
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
dl = data.DataLoader(ds, batch_size=bsz, shuffle=False, num_workers=2, drop_last=True)		

for current_epoch in tqdm(range(1, num_epoch+1)):
	for batch_idx, batch in enumerate(dl):
		current_month, avg_context, high_res_context = batch
		real_labels = to_variable(torch.LongTensor(np.ones(batch_size, dtype = int)), requires_grad = False)
		fake_labels = to_variable(torch.LongTensor(np.zeros(batch_size, dtype = int)), requires_grad = False)
		input = ds.build_input_for_D(current_month, avg_context, high_res_context)
		#move batch to GPU
		input = input.to(device)		
		
		#train Discriminator for k iterations
		for i in xrange(k):
			netD.zero_grad()
			netG.zero_grad()
			outputs = netD(input).squeeze()
			d_real_loss = loss_func(outputs, real_labels)
			z = ds.get_noise(z_shape)
			fake_inputs = netG(z, avg_context, high_res_context)
			outputs = netD(fake_inputs).squeeze()
			d_fake_loss = loss_func(outputs, fake_labels)
			d_loss = -(d_real_loss + d_fake_loss)
			d_loss.backward()
			d_optim.step()
		print("epoch {}, step {},  d loss = {}".format(current_epoch, k,d_loss.item()))
		
		#train Generator
		netD.zero_grad()
		netG.zero_grad()
		
		noise = ds.get_noise()
		g_outputs_fake = netG(noise, avg_context, high_res_context)
		d_input = ds.build_input_for_D(g_outputs_fake, avg_context, high_res_context)
		outputs = netD(d_input).squeeze()
		g_loss = (-1) * loss_func(outputs, fake_labels)
		g_loss.backward()
		#distribute gradient averaging
		average_gradients()
		g_optim.step()
		
		print("epoch {}, g_loss = {}\n".format(current_epoch,g_loss.item()))

