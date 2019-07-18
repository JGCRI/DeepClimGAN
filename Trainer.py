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
from tqdm import tqdm
from torch.autograd import Variable
from DataSampler import DataSampler


#import logger
from sacred import Experiment
from sacred.observers import MongoObserver
import Utils


ex = Experiment('Test experiment')


#MongoDB
#DATABSE_URL = '172.18.65.215'
DATABASE_URL = '172.20.242.12:2701'
DATABASE_NAME = 'climate_gan'
ex.observers.append(MongoObserver.create(url=DATABASE_URL, db_name=DATABASE_NAME))




#sacred configs
@ex.config
def my_config():
	context_length = 5
	lon, lat, t, channels = 128, 256, 30, len(clmt_vars	
	z_shape = 100
	n_days = 32
	apply_norm = True
	batch_size = sys.argv[6]
	lr = 0.0002 #NOTE: this lr is coming from original paper about DCGAN
	l1_lambda = 10
	
	#Percent of data to use, default = use all
	data_pct = 1
	#Percent of data to use for training
	train_pct = 0.7
	data_dir = sys.argv[3]
	num_epoch = int(sys.argv[2])



class Trainer:
	@ex.capture
	def __init__(self):
		self.lon. self.lat, self.context_length, self.channels, self.batch_size = self.set_parameters()
		
	@ex.capture
	def set_parameters(self, lon, lat, context_length, channels, batch_size):
		return lon, lat, context_length, channels, batch_size


	@ex.capture
	def get_exp_info(self, _run):
		exp_id = _run._id
		exp_name = _run.experiment_info["name"]
		return exp_id, exp_name

	@ex.capture
	def run(self):
		"""
		Main routine
		"""

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")		
		#build models
		netD = Discriminator()
		netG = Generator(self.lon, self.lat, self.context_length, self.channels, self.batch_size)
		netD.apply(weights_init)
		netG.apply(weights_init)
		

		
		#create optimizers
		loss_func = torch.nn.CrossEntropyLoss()
		d_optim = torch.optim.Adam(netD.parameters(), lr, [0.5, 0.999])
		g_optim = torch.optim.Adam(netG.parameters(), lr, [0.5, 0.999])
				
		

		#use all possible GPU cores available
		if torch.cuda.device_count() > 1:
    			print("Using ", torch.cuda.device_count(), "GPUs!")
    			netD = nn.DataParallel(netD)
    			netG = nn.DataParallel(netG)
		netD.to(device)
		netG.to(device)

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
	

		#Lists to keep track of gradients
		G_grads, D_grads = [], []
		
		#Training loop
		n_updates = 1
		for current_epoch in range(1, num_epoch + 1):
			#n_updates = 1
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
					#save gradients for D
					d_grad = check_grads(netD, "Discriminator")
					self._run.log_scalar('D_grads', d_grad, n_updates)
					netD.zero_grad()
					
					#concatenate context with the input to feed D
					input = ds.build_input_for_D(current_month, avg_context, high_res_context)
		
					#1A. Train D on real
					outputs = netD(input)
					bsz, ch, h, w, t = outputs.shape
					outputs = outputs.view(bsz, h * w * t)
					d_real_loss = loss_func(outputs, real_labels)
					d_real_loss.backward()	
					#report d_real_loss
					self._run.log_scalar('d_real_loss', d_real_loss.item(), n_updates)
				
					#1B. Train D on fake
					high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)			
					fake_inputs = netG(z, avg_ctxt_for_G, high_res_for_G)
					fake_input_with_ctxt = ds.build_input_for_D(fake_inputs, avg_context, high_res_context)
					
					#feed fake input augmented with the context to D
					outputs = netD(fake_input_with_ctxt.detach()).view(batch_size, h * w * t)
					d_fake_loss = loss_func(outputs, fake_labels)
					d_fake_loss.backward()
					#report d_fake_loss
					self._run.log_scalar('d_fake_loss', d_fake_loss.item(), n_updates)
					
					#Add the gradients from the all-real and all-fake batches	
					d_loss = d_real_loss + d_fake_loss
					#report d_loss
					self._run.log_scalar('d_loss', d_loss.item(), n_updates)
					
					#Update D
					d_optim.step()
					print("epoch {}, update {}, d loss = {:0.18f}, d real = {:0.18f}, d fake = {:0.18f}".format(current_epoch, n_updates, d_loss.item(), d_real_loss.item(), d_fake_loss.item()))
				else:
				
					#2. Train Generator on D's response: maximize log(D(G(z))
					#report grads
					g_grad = check_grads(netG, "Generator")
					self._run.log_scalar('G_grads' g_grad, n_updates)
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
				
					self._run.log_scalar('g_loss', g_loss.item(), n_updates)
					
				n_updates += 1	
	
		losses = D_losses, G_losses
		grads = D_grads, G_grads
		save_results(sys, losses, grads)
	


@ex.main
def main(_run):
	t = Trainer()
	t.run()


if __name__ == "__main__":
	num_experiments = 5
	
	for i in range(num_experiments):
		ex.run()
