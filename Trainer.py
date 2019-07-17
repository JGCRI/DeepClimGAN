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
from DataSampler import DataSampler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
def weights_init(m):
	"""
	Custom weights initialization called on Generator and Discriminator
	All weights are initialized from a zero-centered Normal Distribution
	with std=0.01
	 param: m ()
	return: None
	"""
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
	        m.weight.data.normal_(0.0, 0.01)
	elif classname.find('BatchNorm') != -1:
	        m.weight.data.normal_(1.0, 0.02)
	        m.bias.data.fill_(0)


def to_variable(x, requires_grad = True):
    x = x.to(device)
    return Variable(x, requires_grad)


def check_grads(model, model_name):
	'''
	Intended to be used to check the gradients.
	Use before zeroing gradients out.
	Usage:
	
		check_grads(netG, "Generator")
		netG.zero_grad() )
	'''
	grads = []
	for p in model.parameters():
		if not p.grad is None:
			grads.append(float(p.grad.mean()))
	grads = np.array(grads)
	#if len(grads) > 0:
	#	print(f" Gradients mean, max  for ({model_name}) are ({grads.mean()}, {grads.max()})")
	if len(grads) > 0:
		return grads.mean()
	elif len(grads) == 0:
		return 0
	#if grads.any() and grads.mean() > 100:
	#	print(f"WARNING! gradients mean is over 100 ({model_name})")
	#if grads.any() and grads.max() > 100:
	#	print(f"WARNING! gradients max is over 100 ({model_name})")
	






#Default params
data_dir = '../clmt_data/'
#Percent of data to use for training
train_pct = 0.7
#Percent of data to use at all
data_pct = 1
n_days = 32
#dimension of Gaussian distribution to sample noise from
z_shape = 100
#number of iterations to update Discrimnator (default = 1)
k = 1
#normalize data
apply_norm = True
num_epoch = 5
#self-explanatory
batch_size = 16
lr = 0.0002 #NOTE: this lr is coming from original paper about DCGAN
l1_lambda = 10
lon, lat, t, channels = 128, 256, 30, len(clmt_vars)
context_length = 5


netD = Discriminator()
netG = Generator(lon, lat, context_length, channels, batch_size)
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
sampler = DataSampler(ds, batch_size, context_length, n_days)
b_sampler = data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)
dl = data.DataLoader(ds, batch_sampler=b_sampler, num_workers=0)
dl_iter = iter(dl)



#Lists to keep track of progress:
G_losses = []
D_losses = []

#Lists to keep track of gradients
G_grads = []
D_grads = []

#Training loop
for current_epoch in range(1, num_epoch + 1):
	n_updates = 1
	while True:
		#sample batch
		try:
			batch = next(dl_iter)
		except StopIteration:
			#end of epoch -> shuffle dataset and reinitialize iterator
			sampler.permute()
			print("shuffling batches...\n")
			dl_iter = iter(dl)
			#start new epoch
			break
		
		#unwrap the batch			
		current_month, avg_context, high_res_context = batch["curr_month"], batch["avg_ctxt"], batch["high_res"]
		
		#smoothing labels
		real_labels = to_variable(torch.LongTensor(np.ones(batch_size, dtype=int)), requires_grad = False)
		fake_labels = to_variable(torch.LongTensor(np.zeros(batch_size, dtype = int)), requires_grad = False)
		
		#ship tensors to devices
		current_month = current_month.to(device)
		avg_context = avg_context.to(device)
		high_res_context = high_res_context.to(device)
		
		#sample noise
		z = ds.get_noise(z_shape, batch_size)
		
		#1. Train Discriminator on real+fake: maximize log(D(x)) + log(1-D(G(z))
		if n_updates % 2 == 1:
			d_grad = check_grads(netD, "Discriminator")
			D_grads.append(d_grad)
			netD.zero_grad()
			#concatenate context with the input to feed D
			input = ds.build_input_for_D(current_month, avg_context, high_res_context)

			#1A. Train D on real
			outputs = netD(input)
			bsz, ch, h, w, t = outputs.shape
			outputs = outputs.view(bsz, h * w * t)
			d_real_loss = loss_func(outputs, real_labels)
			d_real_loss.backward()	
		
			#1B. Train D on fake
			high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)			
			fake_inputs = netG(z, avg_ctxt_for_G, high_res_for_G)
			fake_input_with_ctxt = ds.build_input_for_D(fake_inputs, avg_context, high_res_context)
			
			#feed fake input augmented with the context to D
			outputs = netD(fake_input_with_ctxt.detach()).view(batch_size, h * w * t)
			d_fake_loss = loss_func(outputs, fake_labels)
			d_fake_loss.backward()

			#Add the gradients from the all-real and all-fake batches	
			d_loss = d_real_loss + d_fake_loss
			
			#Save loss for plotting later
			D_losses.append(d_loss.item())

			#Update D
			d_optim.step()
			print("epoch {}, update {}, d loss = {:0.18f}, d real = {:0.18f}, d fake = {:0.18f}".format(current_epoch, n_updates, d_loss.item(), d_real_loss.item(), d_fake_loss.item()))
		else:
		
			#2. Train Generator on D's response: maximize log(D(G(z))
			g_grad = check_grads(netG, "Generator")
			G_grads.append(g_grad)
			netG.zero_grad()
			high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)
			g_outputs_fake = netG(z, avg_ctxt_for_G, high_res_for_G)
			d_input = ds.build_input_for_D(g_outputs_fake, avg_context, high_res_context)
			outputs = netD(d_input)
			bsz, c, h, w, t = outputs.shape
			outputs = outputs.view(bsz, h * w * t)
			g_loss = loss_func(outputs, real_labels)#compute loss for G
			g_loss.backward()#only optimize G's parameters
			g_optim.step()
			
			print("epoch {}, update {}, g_loss = {:0.18f}\n".format(current_epoch, n_updates, g_loss.item()))
		
			#Save loss for plotting later:
			G_losses.append(g_loss.item())
	
		n_updates += 1

#Visualize results:
#plt.figure(figsize=(10,5))
#plt.title("Generator and Discriminator Loss During Training")
#plt.plot(G_losses, label="G")
#plt.plot(D_losses, label="D")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()

#save G losses and D losses
import csv
with open('../g_losses.csv','w') as myfile:
	for l in G_losses:
		myfile.write(str(l))
		myfile.write('\n')

with open('../d_losses.csv', 'w') as myfile:
	for l in D_losses:
		myfile.write(str(l))
		myfile.write('\n')


with open('../g_grads.csv', 'w') as myfile:
	for g in G_grads:
		myfile.write(str(g))
		myfile.write('\n')

with open('../d_grads.csv', 'w') as myfile:
	for d in D_grads:
		myfile.write(str(d))
		myfile.write('\n')
