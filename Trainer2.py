import numpy as np
import os
import sys
from NETCDFDataset import NETCDFDataset
from Discriminator import Discriminator
from Generator import Generator
from Normalizer import Normalizer
import torch
import torch.nn as nn
from torch.utils import data
from Constants import clmt_vars
from tqdm import tqdm
from torch.autograd import Variable
from DataSampler import DataSampler
#import chocolate as choco
from Utils import GaussianNoise
#import logger
from sacred import Experiment
from sacred.observers import MongoObserver
import Utils as ut

context_length = 5
lon, lat, t, channels = 128, 256, 30, len(clmt_vars)
z_shape = 100
n_days = 32
apply_norm = True

#Percent of data to use, default = use all
data_pct = 0.2
#Percent of data to use for training
train_pct = 0.7
data_dir = sys.argv[3]

#hyperparamteres TODO:
label_smoothing = False
add_noise_to_real = True
experience_replay = False
batch_size = int(sys.argv[5])
lr = 0.0002 #NOTE: this lr is coming from original paper about DCGAN
l1_lambda = 10
num_epoch = int(sys.argv[2])


class Trainer:
	def __init__(self):
		self.lon, self.lat, self.context_length, self.channels, self.z_shape, self.n_days, self.apply_norm, self.data_dir = lon, lat, context_length, channels, z_shape, n_days, apply_norm, data_dir
		self.label_smoothing, self.add_noise_to_real, self.experience_replay, self.batch_size = label_smoothing, add_noise_to_real, experience_replay, batch_size
		self.lr = lr
		self.num_epoch = num_epoch
		device = self.device = torch.device("cpu")
		#self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	


	def run(self):
		"""
		Main routine
		"""

		#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")		
		device = torch.device("cpu")
		#build models
		netD = Discriminator(self.label_smoothing)
		netG = Generator(self.lon, self.lat, self.context_length, self.channels, self.batch_size)
		netD.apply(ut.weights_init)
		netG.apply(ut.weights_init)
		
		

		#create optimizers
		loss_func = torch.nn.CrossEntropyLoss()
		if self.label_smoothing:
			loss_func = torch.nn.KLDivLoss()
		d_optim = torch.optim.Adam(netD.parameters(), self.lr, [0.5, 0.999])
		g_optim = torch.optim.Adam(netG.parameters(), self.lr, [0.5, 0.999])
				
		
		GPU = False
		#use all possible GPU cores available
		#if GPU and torch.cuda.device_count() > 1:
    		#	print("Using ", torch.cuda.device_count(), "GPUs!")
    		#	netD = nn.DataParallel(netD)
    		#	netG = nn.DataParallel(netG)
		netD.to(device)
		netG.to(device)

		print("Started parsing data...")
		ds = NETCDFDataset(data_dir, data_pct,train_pct)
		
		if self.apply_norm:
			normalizer = Normalizer()
			#normalize training set
			ds.normalized_train = normalizer.normalize(ds, 'train')
		
		
		#Specify that we are loading training set
		sampler = DataSampler(ds, self.batch_size, self.context_length, self.n_days)
		b_sampler = data.BatchSampler(sampler, batch_size=self.batch_size, drop_last=True)
		dl = data.DataLoader(ds, batch_sampler=b_sampler, num_workers=0)
		dl_iter = iter(dl)
		
		D_grads, G_grads = [], []
		D_losses, D_real_losses, D_fake_losses, G_losses = [], [], [], []

		
		#Training loop
		n_updates = 1
		for current_epoch in range(1, self.num_epoch + 1):
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
				
				if self.label_smoothing:
					ts = np.full((self.batch_size), 0.9)
					real_labels = ut.to_variable(torch.FloatTensor(ts), device, requires_grad = False)
				else:
					real_labels = ut.to_variable(torch.LongTensor(np.ones(self.batch_size, dtype=int)), device,requires_grad = False)
									
				fake_labels = ut.to_variable(torch.LongTensor(np.zeros(self.batch_size, dtype = int)), device,requires_grad = False)
				


				#ship tensors to devices
				current_month = current_month.to(device)
				avg_context = avg_context.to(device)
				high_res_context = high_res_context.to(device)
				
				#sample noise
				z = ds.get_noise(self.z_shape, self.batch_size)
				
				#1. Train Discriminator on real+fake: maximize log(D(x)) + log(1-D(G(z))
				if n_updates % 2 == 1:
					#save gradients for D
					d_grad = ut.save_grads(netD, "Discriminator")
					D_grads.append(d_grad)
					netD.zero_grad()
				
					#concatenate context with the input to feed D
					input = ds.build_input_for_D(current_month, avg_context, high_res_context)
					if self.add_noise_to_real:
						add_noise = GaussianNoise(device)
						input = add_noise(input)


					#1A. Train D on real
					outputs = netD(input)
					bsz, ch, h, w, t = outputs.shape
					print(outputs.shape)
					outputs = outputs.view(bsz, h * w * t)
					print(outputs.shape, real_labels.shape)
					d_real_loss = loss_func(outputs, real_labels)
					d_real_loss.backward()	
					#report d_real_loss
					D_real_losses.append(d_real_loss.item())
					#self._run.log_scalar('d_real_loss', d_real_loss.item(), n_updates)
				
					#1B. Train D on fake
					high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)			
					fake_inputs = netG(z, avg_ctxt_for_G, high_res_for_G)
					fake_input_with_ctxt = ds.build_input_for_D(fake_inputs, avg_context, high_res_context)
					
					#feed fake input augmented with the context to D
					outputs = netD(fake_input_with_ctxt.detach()).view(batch_size, h * w * t)
					d_fake_loss = loss_func(outputs, fake_labels)
					d_fake_loss.backward()
					#report d_fake_loss
					#self._run.log_scalar('d_fake_loss', d_fake_loss.item(), n_updates)
					D_fake_losses.append(d_fake_loss.item())
								
					#Add the gradients from the all-real and all-fake batches	
					d_loss = d_real_loss + d_fake_loss
					#report d_loss
					#self._run.log_scalar('d_loss', d_loss.item(), n_updates)
					D_losses.append(d_loss.item())	
					#acc = 
					
					#Update D
					d_optim.step()
					print("epoch {}, update {}, d loss = {:0.18f}, d real = {:0.18f}, d fake = {:0.18f}".format(current_epoch, n_updates, d_loss.item(), d_real_loss.item(), d_fake_loss.item()))
				else:
				
					#2. Train Generator on D's response: maximize log(D(G(z))
					#report grads
					g_grad = ut.save_grads(netG, "Generator")
					#self._run.log_scalar('G_grads', g_grad, n_updates)
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
					G_losses.append(g_loss.item())
					
					#self._run.log_scalar('g_loss', g_loss.item(), n_updates)
					
				n_updates += 1	
	
		#losses = D_losses, D_real_losses, D_fake_losses, G_losses
		#grads = D_grads, G_grads
		#ut.save_results(sys, losses, grads)
	


def main():
	t = Trainer()
	t.run()


main()
