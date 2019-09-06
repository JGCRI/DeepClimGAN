import numpy as np
import os
import sys
from NETCDFDataPartition import NETCDFDataPartition
from Discriminator import Discriminator
from Generator import Generator
from Normalizer import Normalizer
import torch
import torch.nn as nn
from torch.utils import data
from Constants import clmt_vars, GB_to_B
from tqdm import tqdm
from torch.autograd import Variable
from DataSampler import DataSampler
import logging
from sacred import Experiment
from sacred.observers import MongoObserver
import Utils as ut
from Utils import GaussianNoise
from Utils import sort_files_by_size, snake_data_partition
import torch.distributed as dist
import argparse
import logging
import csv


exp_id = 21

ex = Experiment('Experiment ' + str(exp_id) +', pretrain generator')


#MongoDB
DATABASE_URL = "172.18.65.219:27017"
DATABASE_NAME = "climate_gan"

ex.observers.append(MongoObserver.create(url=DATABASE_URL, db_name=DATABASE_NAME))


n_channels = len(clmt_vars)
#sacred configs
@ex.config
def my_config():
	context_length = 5
	lon, lat, t, channels = 128, 256, 30, len(clmt_vars)
	z_shape = 100
	n_days = 32
	apply_norm = True
	
	#hyperparamteres TODO:
	label_smoothing = False
	add_noise = True
	experience_replay = True
	replay_buffer_size = batch_size * 20
	lr = 0.0002 #NOTE: this lr is coming from the original paper about DCGAN
	l1_lambda = 10

class Trainer:
	@ex.capture
	def __init__(self):
		self.lon, self.lat, self.context_length, self.channels, self.z_shape, self.n_days, self.apply_norm, self.data_dir = self.set_parameters()
		self.label_smoothing, self.add_noise, self.experience_replay, self.batch_size, self.lr, self.l1_lambda, self.num_epoch, self.replay_buffer_size, self.report_avg_loss, self.gen_data_dir, self.real_data_dir, self.save_gen_data_update, self.n_data_to_save, self.norms_dir, self.pretrain, self.save_model_dir  = self.set_hyperparameters()
		self.exp_id, self.exp_name, self._run = self.get_exp_info()
		#buffer for expereince replay		
		self.replay_buffer = []
		self.sorted_files = sort_files_by_size(self.data_dir)
		self.world_size = dist.get_world_size()		
		self.partition = snake_data_partition(self.sorted_files, self.world_size)

	@ex.capture
	def set_parameters(self, lon, lat, context_length, channels, z_shape, n_days, apply_norm, data_dir):
		return lon, lat, context_length, channels, z_shape, n_days, apply_norm, data_dir
	
	@ex.capture
	def set_hyperparameters(self, label_smoothing, add_noise, experience_replay, batch_size, lr, l1_lambda, num_epoch, replay_buffer_size, report_avg_loss, gen_data_dir, real_data_dir, save_gen_data_update, n_data_to_save, norms_dir, pretrain, save_model_dir):
		return label_smoothing, add_noise, experience_replay, batch_size, lr, l1_lambda, num_epoch, replay_buffer_size, report_avg_loss, gen_data_dir, real_data_dir, save_gen_data_update, n_data_to_save, norms_dir, pretrain, save_model_dir
	
	@ex.capture
	def get_exp_info(self, _run):
		exp_id = _run._id
		exp_name = _run.experiment_info["name"]
		return exp_id, exp_name, _run

	@ex.capture
	def run(self):
		"""
		Main routine
		"""
		#build models
		netD = Discriminator(self.label_smoothing)
		netG = Generator(self.lon, self.lat, self.context_length, self.channels, self.batch_size)
		netD.apply(ut.weights_init)
		netG.apply(ut.weights_init)		

		#create optimizers
		loss_func = torch.nn.BCELoss()
		mse_loss_func = torch.nn.MSELoss()
		
		if self.label_smoothing:
			loss_func = torch.nn.KLDivLoss()
		
		d_optim = torch.optim.Adam(netD.parameters(), self.lr, [0.5, 0.999])
		g_optim = torch.optim.Adam(netG.parameters(), self.lr, [0.5, 0.999])
				
		
		device = torch.cuda.current_device()
		cpu_dev = torch.device("cpu")
		
		netD.to(device)
		netG.to(device)
		
		comm_size = self.world_size

		#reset the batch size based on the number of processes used
		self.batch_size = self.batch_size // comm_size
		self.replay_buffer_size = 20 * self.batch_size
		n_data_to_save_on_process = self.n_data_to_save // comm_size #should be on CPU
		
		real_data_saved, gen_data_saved = [], []		
		
		rank = dist.get_rank()
		partition = self.partition[rank]
		
		ds = NETCDFDataPartition(partition, self.data_dir)
				
		#Specify that we are loading training set
		sampler = DataSampler(self.batch_size, ds.data, self.context_length, self.n_days)
		b_sampler = data.BatchSampler(sampler, batch_size=self.batch_size, drop_last=True)
		dl = data.DataLoader(ds, batch_sampler=b_sampler)
		dl_iter = iter(dl)
	
		loss_n_updates, report_step = 0, 0
		n_real_saved, n_gen_saved = 0, 0
		dates_mapping = []
		min_loss = float('inf')
			
		#Training loop
		for current_epoch in range(1, self.num_epoch + 1):		
			n_updates = 1
			
			D_epoch_loss, G_epoch_loss = 0, 0

			while True:
				#sample batch
				try:
					batch = next(dl_iter)
				except StopIteration:
					#end of epoch -> shuffle dataset and reinitialize iterator
					sampler.permute()
					dl_iter = iter(dl)
					#start new epoch
					break
		
				if not self.pretrain:
					#report avg_loss per epoch
					if loss_n_updates == report_avg_loss:
						D_epoch_loss = torch.tensor(D_epoch_loss).to(device)
						G_epoch_loss = torch.tensor(G_epoch_loss).to(device)
						all_reduce_dist([D_epoch_loss])
						all_reduce_dist([G_epoch_loss])									
						D_avg_epoch_loss = D_epoch_loss / loss_n_updates
						G_avg_epoch_loss = G_epoch_loss / loss_n_updates
						self._run.log_scalar('D_avg_epoch_loss', D_avg_epoch_loss.item(), report_step)
						self._run.log_scalar('G_avg_epoch_loss', G_avg_epoch_loss.item(), report_step)
						loss_n_updates = 0
						report_step += 1
						D_epoch_loss, G_epoch_loss = 0, 0
					
				#unwrap the batch			
				current_month, avg_context, high_res_context, year_month_date = batch["curr_month"], batch["avg_ctxt"], batch["high_res"], batch["year_month_date"]
				current_month_to_save = current_month
					
		
				if self.label_smoothing:
					ts = np.full((self.batch_size), 0.9)
					real_labels = ut.to_variable(torch.FloatTensor(ts), requires_grad = False)
				else:
					real_labels = ut.to_variable(torch.FloatTensor(np.ones(self.batch_size, dtype=int)),requires_grad = False)
									
				fake_labels = ut.to_variable(torch.FloatTensor(np.zeros(self.batch_size, dtype = int)), requires_grad = False)
				
				real_labels = real_labels.to(device)
				fake_labels = fake_labels.to(device)
				
				#ship tensors to devices
				current_month = current_month.to(device)
				avg_context = avg_context.to(device)
				high_res_context = high_res_context.to(device)
				
				#sample noise
				z = ds.get_noise(self.z_shape, self.batch_size)
				z = z.to(device)
				
				if self.pretrain:
					high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)
					loss = pretrain_G(netG, z, avg_ctxt_for_G, high_res_for_G,current_month, mse_loss_func, comm_size, device)
					
					loss.backward()
					self._run.log_scalar('g_loss {}'.format(loss.item()))
					logging.info("epoch {}, rank {}, update {}, g loss {:0.18f}".format(current_epoch, rank, n_updates, loss.item()))
					average_gradients()
					g_optim.step()
					#if loss.item() < min_loss:
					#	min_loss = min(min_loss, loss.item())
					#	save_model(netG, self.save_model_dir + str(exp_id))
			
				"""
				#1. Train Discriminator on real+fake: maximize log(D(x)) + log(1-D(G(z))
				if n_updates % 2 == 1:
					#save gradients for D
					d_grad = ut.save_grads(netD, "Discriminator")
					self._run.log_scalar('D_grads', d_grad, n_updates)

	
					netD.zero_grad()
					#concatenate context with the input to feed D
					if self.add_noise:
						self.noise = GaussianNoise(device)
						current_month = self.noise(current_month)
					
					input = ds.build_input_for_D(current_month, avg_context, high_res_context)				
					
					#1A. Train D on real
					outputs = netD(input).squeeze()
					d_real_loss = loss_func(outputs, real_labels)
					
					#save real to calculate statistics
					if n_updates >= self.save_gen_data_update and n_real_saved < n_data_to_save_on_process:
						current_month_to_save = current_month_to_save.to(cpu_dev)
						real_data_saved.append(current_month_to_save)
						n_real_saved += self.batch_size
						dates_mapping.append(year_month_date)
						if n_real_saved >= n_data_to_save_on_process:
							save_data(real_data_saved, self.real_data_dir, rank, exp_id, dates_mapping)
							#free buffer
							real_data_saved = []
							dates_mapping = []
					
						
					#report d_real_loss
					self._run.log_scalar('d_real_loss', d_real_loss.item(), n_updates)
				
					#1B. Train D on fake
					high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)			
					
					fake_inputs = netG(z, avg_ctxt_for_G, high_res_for_G)
					
					#save data to calculate statistics
					if n_updates >= self.save_gen_data_update and n_gen_saved < n_data_to_save_on_process:
						fake_inputs = fake_inputs.to(cpu_dev)
						gen_data_saved.append(fake_inputs)
						fake_inputs = fake_inputs.to(device)
						n_gen_saved += self.batch_size
						if n_gen_saved >= n_data_to_save_on_process:
							save_data(gen_data_saved, self.gen_data_dir, rank, exp_id, dates_mapping)
							#free buffer
							gen_data_saved = []
						
						
					if self.add_noise:
						fake_inputs = self.noise(fake_inputs)
					
					fake_input_with_ctxt = ds.build_input_for_D(fake_inputs, avg_context, high_res_context)
					D_input = fake_input_with_ctxt
											
					#feed fake input augmented with the context to D
					if self.experience_replay:
						if current_epoch == 1 and n_updates == 1:
							D_input = fake_input_with_ctxt
						else:
							perm = torch.randperm(self.replay_buffer.shape[0])
							half = self.batch_size // 2
							buffer_idx = perm[:half]
							samples_from_buffer = self.replay_buffer[buffer_idx].to(device)
							perm = torch.randperm(fake_input_with_ctxt.shape[0])
							fake_idx = perm[:half]
							samples_from_G = fake_input_with_ctxt[fake_idx]
							D_input = torch.cat((samples_from_buffer, samples_from_G), dim=0)
								
					outputs = netD(D_input.detach()).squeeze()
					
					d_fake_loss = loss_func(outputs, fake_labels)
					
					#report d_fake_loss
					self._run.log_scalar('d_fake_loss', d_fake_loss.item(), n_updates)
					
					#Add the gradients from the all-real and all-fake batches	
					d_loss = d_real_loss + d_fake_loss
					D_epoch_loss += d_loss.item()
					
					d_loss.backward()
					
					#AVERAGE_GRADIENTS
					average_gradients(netD)
					
					#report d_loss
					self._run.log_scalar('d_loss', d_loss.item(), n_updates)
					
					#Update weights of D
					d_optim.step()
					
					#Update experience replay
					if self.experience_replay:
						#save random 1/2 of batch of generated data
						perm = torch.randperm(fake_input_with_ctxt.shape[0])
						half = self.batch_size // 2
						idx = perm[:half]
						samples_to_buffer = fake_input_with_ctxt[idx].detach().cpu()
						
						if n_updates == 1 and current_epoch == 1:
							#initialize experience replay
							self.replay_buffer = samples_to_buffer
						else:
							#first grow the buffer to the needed buffer size, and then start replacing data
							#replace  1/2 of batch size from the buffer with the newly \
							#generated data
							if self.replay_buffer.shape[0] == self.replay_buffer_size:
								#replace by new generated data
								perm = torch.randperm(self.replay_buffer.shape[0])
								idx = perm[:half]
								self.replay_buffer[idx] = samples_to_buffer
							else:	
								#add new data
								self.replay_buffer = torch.cat((self.replay_buffer, samples_to_buffer),dim=0) 
				
					logging.info("epoch {}, rank {}, update {}, d loss = {:0.18f}, d real = {:0.18f}, d fake = {:0.18f}".format(current_epoch, rank, n_updates, d_loss.item(), d_real_loss.item(), d_fake_loss.item()))
				else:
				
					#2. Train Generator on D's response: maximize log(D(G(z))
					#report grads
					g_grad = ut.save_grads(netG, "Generator")
					self._run.log_scalar('G_grads', g_grad, n_updates)
					netG.zero_grad()
					
					high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_context, high_res_context)
					g_outputs_fake = netG(z, avg_ctxt_for_G, high_res_for_G)
					d_input = ds.build_input_for_D(g_outputs_fake, avg_context, high_res_context)
					outputs = netD(d_input).squeeze()
					g_loss = loss_func(outputs, real_labels)#compute loss for G
					g_loss.backward()
					
					#save generated data
					if n_updates >= self.save_gen_data_update and n_gen_saved < n_data_to_save_on_process:
						g_outputs_fake = g_outputs_fake.to(cpu_dev)
						gen_data_saved.append(g_outputs_fake)
						g_outputs_fake = g_outputs_fake.to(device)
						n_gen_saved += self.batch_size
						dates_mapping.append(year_month_date)
						if n_gen_saved >= n_data_to_save_on_process:
							save_data(gen_data_saved, self.gen_data_dir, rank, exp_id, dates_mapping)
							#free buffer
							gen_data_saved = []
					#save real data
					if n_updates >= self.save_gen_data_update and n_real_saved < n_data_to_save_on_process:
						current_month_to_save = current_month_to_save.to(cpu_dev)
						real_data_saved.append(current_month_to_save)
						n_real_saved += self.batch_size
						dates_mapping.append(year_month_date)
						if n_real_saved >= n_data_to_save_on_process:
							save_data(real_data_saved, self.real_data_dir, rank, exp_id, dates_mapping)
							real_data_saved = []
					
					#average gradients
					average_gradients(netG)
				
					#update weights of G				
					g_optim.step()
					
					G_epoch_loss += g_loss
					logging.info("epoch {}, rank {}, update {}, g_loss = {:0.18f}\n".format(current_epoch, rank, n_updates, g_loss.item()))
					self._run.log_scalar('g_loss', g_loss.item(), n_updates)
				"""
					
				n_updates += 1	
				#loss_n_updates += 1

def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
		param.grad.data /= size

def all_reduce_dist(sums):
	for sum in sums:
		dist.all_reduce(sum, op=dist.ReduceOp.SUM)



def pretrain_G(netG, z, avg_ctxt_for_G, high_res_for_G,current_month_batch, mse_loss_func, comm_size, device):
	fake_outputs = netG(z, avg_ctxt_for_G, high_res_for_G)

	#compute targets
	map_mean_target = get_mean_map(current_month_batch, comm_size)
	map_std_target = get_std_map(current_month_batch, comm_size)
	
	#compute stat for fake
	batch_fake_outputs_mean = get_mean_map(fake_outputs, comm_size)
	batch_fake_outputs_std = get_std_map(fake_outputs, comm_size)
	loss = mse_loss_func(map_mean_target, batch_fake_outputs_mean) + mse_loss_func(map_std_target, batch_fake_outputs_std) + get_tas_zero_fraq(fake_outputs)
	return loss


def get_mean_map(batch, comm_size):
	#merge batches along days axes
	bsz = batch.shape[0]	
	tsr = batch.permute(1,2,3,0,4).contiguous()
	tsr = tsr.view(n_channels, 128,256,32 * bsz) #7 x 128 x 256 x bsz*32
	#sum along days dimension
	sum_tsr = tsr.sum(-1)
	all_reduce_sum(sum_tsr)
	mean_tsr = sum_tsr / (bsz * 32 * comm_size)
	return mean_tsr
	

def get_std_map(batch, comm_size):
	bsz = batch.shape[0]
	tsr = batch.permute(1,2,3,0,4).contiguous()
	tsr = tsr.view(n_channels, 128, 256, 32 * bsz) #7 x 128 x 256 x bsz*32
	
	#sum along days in the batch on the current process
	sum_tsr = tsr.sum(-1)
	#summ along days in the batch on all the processes
	all_reduce_sum(sum_tsr)
	mean_tsr = sum_tsr / (bsz * 32 * comm_size)
	mean_tsr.unsqueeze_(-1)#7 x 128 x 256 x 1
	mean_tsr = mean_tsr.expand(7,128,256,32*bsz) #7 x 128 x 256 x 64
	#calc sq diff on current batch on the process and sum them up along days dimension
	sq_diff = ((tsr - mean_tsr) ** 2).sum(-1)
	#calc sq diffs for all the tensors
	all_reduce_sum(sq_diff)
	logging.info("sq diff size {} ".format(sq_diff.shape))
	std = torch.sqrt(sq_diff / (bsz * 32 * comm_size))
	logging.info("std tsr for tas{} ".format(std_tsr[1]))
	return std_tsr

def get_tas_zeros_fraq(batch):
	bsz = batch.shape[0]
	tsr = batch.permute(1,2,3,0,4).contiguous()
	tsr = tsr.view(n_channels, 128, 256, 32 * bsz)
	tas = tsr[clmt_vars.keys().index('tas')]
	tasmin = tsr[clmt_vars.keys().index('tasmin')]
	tasmax = tsr[clmt_vars.keys().index('tasmax')]
	
	res = tas >= tasmin and tas <= tasmax
	res = res.reshape(-1)
	logging.info("tas in range {}".format(res))
	tasmin = tasmin.reshape(-1)
	tas_shape = tasmin.shape
	target = torch.ones(tas_shape)
	target.fill_(True)
	target.cuda()

	zeros = (target != res)
	#sum all zeros on all processes
	zeros_ts = torch.tensor(zeros.shape[0])
	logging.info("zero ts on one process {}".format(zeros_ts))
	zeros_ts.cuda()
	all_reduce_sum(zeros_ts)
	logging.info("zeros ts on all processes {}".format(zero_ts))
	#sum all sizes of tensors
	tas_shape_ts = torch.tensor(tas_shape)
	logging.info("tas shape on one process {}".format(tas_shape_ts))
	tas_shape_ts.cuda()
	all_reduce_sum(tas_shape_ts)
	logging.info("tas shape on all processes {}".format(tas_shape_ts))
	zero_fraq = zeros_ts.item() / tas_shape_ts.item()
	logging.info("zero_fraq {}".format(zero_fraq))
	return zero_fraq

	
def save_model(netG, dir):
	model.save_state_dict(dir + '/model.pt')
	return


def all_reduce_sum(tsr):
	dist.all_reduce(tsr,op=dist.ReduceOp.SUM)
	return

	
def save_data(months_to_save, save_dir, process_rank, exp_id, dates_mapping):
	#Tensor shape is batch_size x 7 x 128 x 256 x 32
	 
	logging.info("Saving data to {}".format(save_dir))
	
	batch_size = months_to_save[0].shape[0]
	n_channels = months_to_save[0].shape[1]
	n_months = len(months_to_save)
	#logging.info("tensor shape {}".format(months_to_save[0].shape))
	#logging.info("batch_size {}, n_channels {}, n_months {}".format(batch_size, n_channels, n_months))
		
	tensor = torch.cat(months_to_save, dim=0)#resulting tensor is batch_size*n_months x 7 x 128 x 256 x 32
	tensor = tensor.permute(1,2,3,0,4).contiguous()
	tensor = tensor.view(n_channels, 128, 256, n_months * 32 * batch_size)

	ts_name = os.path.join(save_dir + "exp_" + str(exp_id) + "/", str(process_rank))
	torch.save(tensor, ts_name + ".pt")
	
	logging.info("Data is saved to {}".format(save_dir))

	#TODO save dates mapping
	fname = os.path.join(save_dir + "exp_" + str(exp_id) + "/", str(process_rank) + "_dates_mapping.csv")
	with open(fname, 'w+') as of:
		csv_writer = csv.writer(of, delimiter=',')
		#write date line by line (each date consists of N months, where N = batch size per process)
		for date in dates_mapping: 
			csv_writer.writerow(date)
	return



@ex.main
def main(_run):
	t = Trainer()
	t.run()	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--local_rank', type=int)
	parser.add_argument('--num_epoch', type=int)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--data_dir', type=str)
	parser.add_argument('--report_loss', type=int)	
	parser.add_argument('--gen_data_dir',type=str)
	parser.add_argument('--real_data_dir', type=str)
	parser.add_argument('--save_gen_data_update',type=int)
	parser.add_argument('--n_data_to_save', type=int)
	parser.add_argument('--norms_dir', type=str)
	parser.add_argument('--pretrain', type=int)
	parser.add_argument('--save_model_dir', type=str)
	

	args = parser.parse_args()

	lrank = args.local_rank
	num_epoch = args.num_epoch
	batch_size = args.batch_size
	data_dir = args.data_dir
	report_avg_loss = args.report_loss	
	gen_data_dir = args.gen_data_dir
	real_data_dir = args.real_data_dir
	save_gen_data_update = args.save_gen_data_update
	n_data_to_save = args.n_data_to_save
	norms_dir = args.norms_dir
	pretrain=args.pretrain
	save_model_dir = args.save_model_dir
	
	print(f'localrank: {lrank}	host: {os.uname()[1]}')
	torch.cuda.set_device(lrank)
	dist.init_process_group('nccl', 'env://')
	

	ex.add_config(
		num_epoch=num_epoch,
		batch_size=batch_size,
		data_dir=data_dir,
		report_avg_loss=report_avg_loss,
		gen_data_dir=gen_data_dir,
		real_data_dir=real_data_dir,
		save_gen_data_update=save_gen_data_update,
		n_data_to_save=n_data_to_save, 
		norms_dir=norms_dir,
		pretrain=pretrain,
		save_model_dir=save_model_dir)
	
	ex.run()
