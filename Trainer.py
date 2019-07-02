import numpy as np
import os
import sys
from NETCDFDataset import NETCDFDataset
from Discriminator import Discriminator
from Generator import Generator
from Normalizer import Normalizer
import torch
from torch.utils import data 
import torch.distributed as dist
from torch.multiprocessing import Process
from Utils import Utils



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



netD = Discriminator()
netG = Generator()
netD.apply(weights_init)
netG.apply(weights_init)

#use all possible GPU cores available
if torch.cuda.device.count() > 1:
	print("Using ", torch.device.count(), "GPUs!")
	netD = nn.DataParallel(netD)
	netG = nn.DataParallel(netG)
netD.to(device)
netG.to(device)


loss_func = torch.nn.CrossEntropyLoss()
d_optim = torch.optim.Adam(netD.parameters(), lr, [0.5, 0.999])
g_optim = torch.optim.Adam(netG.parameters(), lr, [0.5, 0.999])


print("Started parsing data...")
ds = NETCDFDataset(data_dir, train_pct)

ds_len = ds.__len__()
train_len = ds.train_len
utils = Utils()
if apply_norm:
	normalizer = Normalizer()
	#compute normalizatio values based on the train set only
	normalizer.normalize(train_len, data_dir, utils, ds.files_to_idx)

#apply lazy loading
print(ds.shape)
dl = data.DataLoader(dataset=ds[:,:,0:ds.train_len,:], batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)	
	
for current_epoch in tqdm(range(1, num_epoch+1)):
	for batch_idx, batch in enumerate(dl):
		input, current_month = batch
		real_labels = to_variable(torch.LongTensor(np.ones(batch_size, dtype = int)), requires_grad = False)
		fake_labels = to_variable(torch.LongTensor(np.zeros(batch_size, dtype = int)), requires_grad = False)
		#move batch to GPU
		input = input.to(device)		
		
		#train Discriminator for k iterations
		for i in xrange(k):
			netD.zero_grad()
			netG.zero_grad()
			outputs = netD(input).squeeze()
			d_real_loss = loss_func(outputs, real_labels)
			z = ds.get_noise(z_shape)
			fake_inputs = netG(z)
			outputs = netD(fake_inputs).squeeze()
			d_fake_loss = loss_func(outputs, fake_labels)
			d_loss = d_real_loss + d_fake_loss
			d_loss.backward()
			d_optim.step()
			print("epoch {}, step {},  d loss = {}".format(current_epoch, k,d_loss.item()))
		
		
		#train Generator
		netD.zero_grad()
		netG.zero_grad()
		
		noise = ds.get_noise()
		g_outputs_fake = netG(noise)
		outputs = netD(g_outputs_fake).squeeze()
		g_loss = loss_func(outputs, fake_labels)
		g_loss.backward()
		g_optim.step()
		
		print("epoch {}, g_loss = {}\n".format(current_epoch,g_loss.item()))



#if __name == "__main__":
#	#Number of nodes we want to use 
#	size = 2
#	processes = []
#	for rank in range(size):
#		p = Process(target=init_processes, args=(rank, size, run))
#		p.start()
#		processes.append(p)
#
#	for p in processes:
#		p.join()
#
