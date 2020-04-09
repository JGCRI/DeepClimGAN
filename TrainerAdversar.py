import numpy as np
import gc
import os
import sys
from DeepClimGAN_Alt.NETCDFDataPartition import NETCDFDataPartition
from DeepClimGAN_Alt.Discriminator_uncond import Discriminator
from DeepClimGAN_Alt.Generator_alt import Generator
#from DeepClimGAN_Alt.Generator_uncond import Generator
#from DeepClimGAN_Alt.Generator import Generator
#from DeepClimGAN_Alt.Discriminator import Discriminator
#from Generator_alt import Generator
from Normalizer import Normalizer
import torch
import torch.nn as nn
from torch.utils import data
from Constants import clmt_vars, GB_to_B
from torch.autograd import Variable
from DataSampler import DataSampler
import logging
from sacred import Experiment
from sacred.observers import MongoObserver
import Utils as ut
from Utils import GaussianNoise
from Utils import sort_files_by_size, snake_data_partition
import argparse
import logging
import csv
from random import randint

exp_id = 114
ex = Experiment('Exp 114, adv training of a model, where G is cond and D uncond, added mean loss only to G update, using model from exp_107, saving data after 10k updates')


#MongoDB
DATABASE_URL = "172.18.65.219:27017"
DATABASE_NAME = "climate_gan"

ex.observers.append(MongoObserver.create(url=DATABASE_URL, db_name=DATABASE_NAME))
n_channels = len(clmt_vars)

#sacred configs
@ex.config
def my_config():
    context_length = 5
    lon, lat, t, channels = 128, 256, 32, len(clmt_vars)
    n_days = 32
    apply_norm = True
    #hyperparamteres
    #label_smoothing = False
    add_noise = False
    experience_replay = False
    replay_buffer_size = batch_size * 20
    G_lr = 0.0001 #NOTE: this lr is coming from the original paper about DCGAN
    D_lr = 0.00001
    l1_lambda = 10
    lambda1 = 1
    lambda2 = 1
    lambda3 = 1
    lambdas = (lambda1, lambda2, lambda3) 
    alpha = randint(2, 5)
    beta = randint(2, 10)

class TrainerAdversar:
    @ex.capture
    def __init__(self):
        self.lon, self.lat, self.context_length, self.channels,  self.apply_norm, self.data_dir, self.lambdas, self.alpha, self.beta = self.set_parameters()
        self.add_noise, self.experience_replay, self.batch_size, self.G_lr, self.D_lr,self.l1_lambda, self.num_epoch, self.replay_buffer_size, self.report_avg_loss, self.gen_data_dir, self.real_data_dir, self.save_gen_data_update, self.n_data_to_save, self.norms_dir,self.save_model_dir, self.z_shape, self.pretrained_model, self.n_generate_for_one_z, self.dir_to_save_z_realizations, self.num_smoothing_conv_layers, self.last_layer_size, self.n_days, self.label_smoothing, self.train_G_with_context, self.train_D_with_lowres_context, self.train_D_with_highres_context,self.start_lr, self.end_lr, self.add_mean_loss, self.partition_start = self.set_hyperparameters()
        self.exp_id, self.exp_name, self._run = self.get_exp_info()
        #buffer for expereince replay        
        self.replay_buffer = []
        self.is_autoencoder = False
        self.sorted_files = sort_files_by_size(self.data_dir)
        self.partition = snake_data_partition(self.sorted_files, 1) # WARNING: don't hardcode this here
        self.save_model = self.save_model_dir + 'exp_' + str(exp_id) + '/'


    @ex.capture
    def set_parameters(self, lon, lat, context_length, channels, apply_norm, data_dir, lambdas, alpha, beta):
        return lon, lat, context_length, channels, apply_norm, data_dir, lambdas, alpha, beta
    
    @ex.capture
    def set_hyperparameters(self, add_noise, experience_replay, batch_size, G_lr, D_lr,l1_lambda, num_epoch, replay_buffer_size, report_avg_loss, gen_data_dir, real_data_dir, save_gen_data_update, n_data_to_save, norms_dir, save_model_dir, z_shape, pretrained_model, n_generate_for_one_z, dir_to_save_z_realizations, num_smoothing_conv_layers, last_layer_size, n_days, label_smoothing, train_G_with_context, train_D_with_lowres_context, train_D_with_highres_context,start_lr, end_lr, add_mean_loss, partition_start):
        return add_noise, experience_replay, batch_size, G_lr,D_lr, l1_lambda, num_epoch, replay_buffer_size, report_avg_loss, gen_data_dir, real_data_dir, save_gen_data_update, n_data_to_save, norms_dir, save_model_dir, z_shape, pretrained_model, n_generate_for_one_z, dir_to_save_z_realizations, num_smoothing_conv_layers, last_layer_size, n_days, label_smoothing, train_G_with_context, train_D_with_lowres_context, train_D_with_highres_context,start_lr, end_lr, add_mean_loss, partition_start
    
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
        netD = Discriminator(self.label_smoothing, self.is_autoencoder, self.z_shape)
        #netD.apply(ut.weights_init)
       
        netG = Generator(self.lon, self.lat, self.context_length, n_channels, self.batch_size, self.z_shape, self.last_layer_size)        
        #netG.apply(ut.weights_init)

        paramsG = torch.load(self.pretrained_model + "netG.pt")
        netG.load_state_dict(paramsG)
  
        paramsD = torch.load(self.pretrained_model + "netD.pt")
        netD.load_state_dict(paramsD)


        #create optimizers
        loss_func = torch.nn.BCEWithLogitsLoss()
        mse_loss_func = torch.nn.MSELoss()
        
        if self.label_smoothing:
            loss_func = torch.nn.KLDivLoss()
        
        d_optim = torch.optim.AdamW(netD.parameters(), self.G_lr)
        g_optim = torch.optim.AdamW(netG.parameters(), self.D_lr)
        
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        cpu_dev = torch.device("cpu")
        
        netD.to(device0)
        netG.to(device1)
        

        #reset the batch size based on the number of processes used
        self.total_batch_size  = self.batch_size        
        self.actual_batch_size = 2 # WARNING: DON'T HARDCODE THIS   #self.batch_size // comm_size
        self.num_workers       = self.batch_size // self.actual_batch_size


        self.replay_buffer_size = 20 * self.total_batch_size
        n_data_to_save_on_process = self.n_data_to_save #// comm_size #should be on CPU
        
        real_data_saved, gen_data_saved = [], []        

        rank = 0
        partition = self.partition[rank] #WARNING: testing using just 3 files from a dataset
        partition = partition[self.partition_start:(len(partition) // 2)]#warning take just a half of a data
        #partition = partition[:1]
        logging.info("partition {}".format(partition))
        ds = NETCDFDataPartition(partition, self.data_dir, self.lat, self.lon, 2, device0) # 2 for number of channels in low-res context
        logging.info("finished loading partition")
         
        if n_generate_for_one_z > 0:
            self.total_batch_size = 2

        sampler = DataSampler(self.total_batch_size, ds.data, self.context_length, self.n_days)
        b_sampler = data.BatchSampler(sampler, batch_size=self.total_batch_size, drop_last=True)
        dl = data.DataLoader(ds, batch_sampler=b_sampler)
        dl_iter = iter(dl)
    
        loss_n_updates, report_step = 0, 0
        n_real_saved, n_gen_saved = 0, 0
        dates_mapping = []
        min_loss = float('inf')
        fake_input_with_ctxt = None

        #self.num_workers = 1 #HARDCODED VALUE            
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

                #unwrap the batch            
                current_month, avg_context, high_res_context, year_month_date = batch["curr_month"], batch["avg_ctxt"], batch["high_res"], batch["year_month_date"]
                current_month_to_save = current_month

                #for confusion matrix for D update and G updates
                total_TP, total_FN, total_FP, total_TN = 0, 0, 0, 0
                total_TN_for_G_update, total_FP_for_G_update = 0, 0

                if self.label_smoothing:
                    ts = np.full((self.total_batch_size), 0.9)
                    real_labels = ut.to_variable(torch.FloatTensor(ts), requires_grad = False)
                else:
                    real_labels = ut.to_variable(torch.FloatTensor(np.ones(self.total_batch_size, dtype=int)),requires_grad = False)
                                    
                fake_labels = ut.to_variable(torch.FloatTensor(np.zeros(self.total_batch_size, dtype = int)), requires_grad = False)
                

                #sample noise
                if n_generate_for_one_z > 0:
                     self.batch_size = 2                

                z = ds.get_noise(self.z_shape, self.batch_size)
                real_labelss = list(torch.chunk(real_labels,self.num_workers))
                fake_labelss = list(torch.chunk(fake_labels,self.num_workers))
                if self.add_noise: # WARNING: may need to be applied to each month in current_months
                    self.noise = GaussianNoise()

                current_months = list(torch.chunk(current_month,self.num_workers))
                avg_contexts = list(torch.chunk(avg_context,self.num_workers))
                high_res_contexts = list(torch.chunk(high_res_context,self.num_workers))
                initial_z = z
                z = list(torch.chunk(z, self.num_workers))
                fake_outputs = []

                #GAN training

                N_real = N_fake = 32.0 #assume batch size is 32
                
                #1. Train Discriminator on real+fake: maximize log(D(x)) + log(1-D(G(z))
                #update D every second and every third time
                #if n_updates % 2 == 0:
         
                #if acc on real is less than 50 -> train F for 20 iterations
                train_D = 5 #hardcoded!
                train_only_D = 0
                count_train_only_D = 0

                if train_only_D or n_updates % 2 == 0:
                    if train_only_D:
                        count_train_only_D += 1
                        if count_train_only_D == train_D:
                            count_train_only_D = 0
                            train_only_D = 0 #reset counter

                    #save gradients for D
                    d_grad = ut.save_grads(netD, "Discriminator")
                    self._run.log_scalar('D_grads', d_grad)
                    logging.info("D gradients: {} ".format(d_grad))  

                    netD.zero_grad()

                    for worker in range(self.num_workers):
                        #ship to GPU
                        current_months[worker] = current_months[worker].to(device0)
                        avg_contexts[worker] = avg_contexts[worker].to(device0)
                        high_res_contexts[worker] = high_res_contexts[worker].to(device0)
                        real_labelss[worker] = real_labelss[worker].to(device0)
                        fake_labelss[worker] = fake_labelss[worker].to(device0)
                        
                        if self.add_noise:
                            current_months[worker] = self.noise(current_months[worker], device0)	


                        #feed real and real			
                        if self.train_D_with_lowres_context or self.train_D_with_highres_context:
                            input = ds.build_input_for_D(current_months[worker], current_months[worker], avg_contexts[worker], high_res_contexts[worker], self.train_D_with_lowres_context, self.train_D_with_highres_context)
                        else:
                            input = current_months[worker]


                        #1A. Train D on real
                        outputs = netD(input).squeeze()
                        d_real_loss = loss_func(outputs, real_labelss[worker])

                        #report d_real_loss
                        self._run.log_scalar('d_real_loss', d_real_loss.item(), n_updates) 
                        
                        #calculate for confusion matrix       
                        TP, FN = get_confusion_matrix(outputs, real_labelss[worker], self.actual_batch_size, "real")
                        total_TP += TP
                        total_FN += FN

                        #1B. Train D on fake
                        if self.train_G_with_context:
                            high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_contexts[worker], high_res_contexts[worker])
                            fake_inputs = netG(z[worker].to(device1), avg_ctxt_for_G.to(device1), high_res_for_G.to(device1))
                        else:
                            fake_inputs = netG(z[worker].to(device1), None, None)
		
	
                        if self.add_noise:
                            fake_inputs = self.noise(fake_inputs, device1)
       
                        #feed fake and real
                        if self.train_D_with_lowres_context or self.train_D_with_highres_context:
                            fake_input_with_ctxt = ds.build_input_for_D(fake_inputs.to(device0), current_months[worker], avg_contexts[worker].to(device0), high_res_contexts[worker].to(device0), self.train_D_with_lowres_context, self.train_D_with_highres_context)
                            D_input = fake_input_with_ctxt
                        else:
                            D_input = fake_inputs.to(device0)


                        #feed fake input augmented with the context to D
                        if self.experience_replay:
                            if current_epoch == 1 and n_updates == 1:
                                D_input = fake_input_with_ctxt
                            else:
                                perm = torch.randperm(self.replay_buffer.shape[0])
                                half = self.actual_batch_size // 2
                                buffer_idx = perm[:half]
                                samples_from_buffer = self.replay_buffer[buffer_idx].to(device)
                                perm = torch.randperm(fake_input_with_ctxt.shape[0])
                                fake_idx = perm[:half]
                                samples_from_G = fake_input_with_ctxt[fake_idx]
                                D_input = torch.cat((samples_from_buffer, samples_from_G), dim=0)
                                    
                        outputs = netD(D_input.detach()).squeeze()
                        d_fake_loss = loss_func(outputs, fake_labelss[worker])
                        self._run.log_scalar('d_fake_loss', d_fake_loss.item(), n_updates)
                        
                        TN, FP = get_confusion_matrix(outputs, fake_labelss[worker], self.actual_batch_size, "fake")
                        total_TN += TN
                        total_FP += FP


                        #Add the gradients from the all-real and all-fake batches    
                        d_loss = d_real_loss + d_fake_loss
                        if self.add_mean_loss:
       	       	       	    d_loss += 0.9 * mean_loss(outputs, current_months[worker])

                        D_epoch_loss += d_loss.item()
                        
                        d_loss.backward()
                        #report d_loss
                        self._run.log_scalar('d_loss', d_loss.item(), n_updates)
                        
                    #Update weights of D
                    d_optim.step()
                    
                    #Update experience replay
                    if self.experience_replay:
                        #save random 1/2 of batch of generated data
                        perm = torch.randperm(fake_input_with_ctxt.shape[0])
                        half = self.actual_batch_size // 2
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
                
                    logging.info("epoch {}, rank {}, update {}, d loss = {:0.30f}, d real = {:0.30f}, d fake = {:0.30f}".format(current_epoch, rank, n_updates, d_loss.item(), d_real_loss.item(), d_fake_loss.item()))
                    d_acc_real = (total_TP / 32.0) * 100
                    d_acc_fake = (total_TN / 32.0) * 100 
                    
                    logging.info("TP = {}, FN = {}, FP = {}, TN = {}".format(total_TP, total_FN, total_FP, total_TN))
                    logging.info("D_acc_real = {:0.3f}, D_acc_fake = {:0.3f}".format(d_acc_real, d_acc_fake))
                else:
                

                    #2. Train Generator on D's response: maximize log(D(G(z))
                    #report grads
                    g_grad = ut.save_grads(netG, "Generator")
                    self._run.log_scalar('G_grads', g_grad)
                    logging.info("G gradients: {}".format(g_grad))
                    netG.zero_grad()
                    netD.zero_grad()                    


                    d_acc_real = 0
                    for worker in range(self.num_workers):
                        avg_contexts[worker] = avg_contexts[worker].to(device1)
                        high_res_contexts[worker] = high_res_contexts[worker].to(device1)
                        real_labelss[worker] = real_labelss[worker].to(device0)
                        z[worker] = z[worker].to(device1)
                        current_months[worker] = current_months[worker].to(device0)
                        fake_labelss[worker] = fake_labelss[worker].to(device0)

                        if self.train_G_with_context:
                            high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_contexts[worker], high_res_contexts[worker])
                        else:
                            high_res_for_G, avg_ctxt_for_G = None, None                        

                        g_outputs_fake = netG(z[worker], avg_ctxt_for_G, high_res_for_G)

                        #ship to device 0 for D
                        g_outputs_fake = g_outputs_fake.to(device0)
                        avg_contexts[worker] = avg_contexts[worker].to(device0)
                        high_res_contexts[worker] = high_res_contexts[worker].to(device0)
           
                        #feed fake and real
                        if self.train_D_with_lowres_context or self.train_D_with_highres_context:
                            d_input = ds.build_input_for_D(g_outputs_fake, current_months[worker], avg_contexts[worker], high_res_contexts[worker], self.train_D_with_lowres_context, self.train_D_with_highres_context)
                        else:
                            d_input = g_outputs_fake
                        
                        outputs = netD(d_input).squeeze()
                        g_loss = loss_func(outputs, real_labelss[worker])#compute loss for G
                        if self.add_mean_loss:
                            g_loss += 0.9 * mean_loss(outputs, current_months[worker])

		
                        #compute additional loss for precipitation for G
                        #g_loss_additional = ut.generators_additional_term(g_outputs_fake, current_months[worker], self.alpha, self.beta, self.lambdas)
                        #g_loss  = g_loss + g_loss_additional
                        
                        TN, FP = get_confusion_matrix(outputs, fake_labelss[worker], self.actual_batch_size, "fake")                 
                        total_TN_for_G_update += TN
                        total_FP_for_G_update += FP

                        #update G
                        g_loss.backward()

                        #save generated data
                        if n_updates >= self.save_gen_data_update and n_gen_saved < n_data_to_save_on_process:
                            g_outputs_fake = g_outputs_fake.to(cpu_dev)
                            gen_data_saved.append(g_outputs_fake)
                            n_gen_saved += self.actual_batch_size
                            dates_mapping.append(year_month_date)
                            if n_gen_saved == n_data_to_save_on_process:
                                save_data(gen_data_saved, self.gen_data_dir, 0, exp_id, dates_mapping) # WARNING: hardcoded rank to 0
                                #free buffer
                                save_model(netG,netD, self.save_model)
                                gen_data_saved = []
                      
                        #save real data
                        if n_updates >= self.save_gen_data_update and n_real_saved < n_data_to_save_on_process:
                            current_months[worker] = current_months[worker].to(cpu_dev)
                            real_data_saved.append(current_months[worker])
                            n_real_saved += self.actual_batch_size
                            dates_mapping.append(year_month_date)
                            if n_real_saved == n_data_to_save_on_process:
                                save_data(real_data_saved, self.real_data_dir, 0, exp_id, dates_mapping) # WARNING: hardcoded rank to 0
                                real_data_saved = []
                        
                    #update weights of G
                    g_optim.step()
                    
                    G_epoch_loss += g_loss
                    logging.info("epoch {}, rank {}, update {}, g_loss = {:0.30f}\n".format(current_epoch, rank, n_updates, g_loss.item()))
                    self._run.log_scalar('g_loss', g_loss.item(), n_updates)
                    d_acc_real = (total_TN_for_G_update / 32.0) * 100.0
                    logging.info("D_acc_fake = {:0.3f}, TN = {}, FP = {}".format(d_acc_real, total_TN_for_G_update, total_FP_for_G_update))

                n_updates += 1    
                loss_n_updates += 1


def log_losses(run, losses, update):
    mean_losses = losses["mean_losses"]
    std_losses = losses["std_losses"]
    rhs_loss = losses["rhs_loss"][0]
    tas_loss = losses["tas_loss"][0]
    total_loss = losses["total_loss"][0]

    for i in range(len(mean_losses)):
        run.log_scalar('loss_mean_' + str(i), mean_losses[i].item(), update)

    for i in range(len(std_losses)):
        run.log_scalar('loss_std_' + str(i), std_losses[i].item(), update)

    run.log_scalar('rhs_loss', rhs_loss.item(), update)
    run.log_scalar('tas_loss', tas_loss.item(), update)
    run.log_scalar('total_loss', total_loss.item(), update)

def get_confusion_matrix(data, labels, batch_size_for_one_worker, data_type):
    """
    data: outputs generated from real or fake data
    size of the each tensor: (batch_size / num_workers) x Channels x H x W x T
    data_type: real or fake

    """
    correct = 0
    if data_type == "real":
        correct = torch.sum((data >= 0.5).float() == labels)
    else:
        correct = torch.sum((data < 0.5).float() == labels)

    correct = int(correct.item())
    incorrect = batch_size_for_one_worker - correct
    return correct, incorrect



def mean_loss(generated, real):
    """
    generated: tensor B_SIZE x n_channels x H x W x T
    real: tensor B_SIZE x n_channels x H x W x T

    """
    #compute mean map across the time:
    gen_mean_map = generated.mean(-1)
    real_mean_map = real.mean(-1)
    return (generated.mean() - real.mean()) ** 2   
    

    
def save_model(netG, netD, dir):
    logging.info("saving the models")
    torch.save(netG.state_dict(), dir + 'netG.pt')
    torch.save(netD.state_dict(), dir + 'netD.pt')
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
    t = TrainerAdversar()
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
    parser.add_argument('--save_model_dir', type=str)
    parser.add_argument('--z_shape', type=int)
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--n_generate_for_one_z', type=int)
    parser.add_argument('--dir_to_save_z_realizations',type=str)
    parser.add_argument('--num_smoothing_conv_layers', type=int)
    parser.add_argument('--last_layer_size', type=int)
    parser.add_argument('--n_days',type=int)
    parser.add_argument('--label_smoothing', type=int)
    parser.add_argument('--train_G_with_context', type=int)
    parser.add_argument('--train_D_with_lowres_context',type=int)
    parser.add_argument('--train_D_with_highres_context', type=int)  
    parser.add_argument('--start_lr', type=float)
    parser.add_argument('--end_lr', type=float) 
    parser.add_argument('--add_mean_loss', type=int)
    parser.add_argument('--partition_start',type=int)

    args = parser.parse_args()

    num_epoch = args.num_epoch
    batch_size = args.batch_size
    data_dir = args.data_dir
    report_avg_loss = args.report_loss    
    gen_data_dir = args.gen_data_dir
    real_data_dir = args.real_data_dir
    save_gen_data_update = args.save_gen_data_update
    n_data_to_save = args.n_data_to_save
    norms_dir = args.norms_dir
    save_model_dir = args.save_model_dir
    z_shape = args.z_shape
    pretrained_model=args.pretrained_model
    n_generate_for_one_z = args.n_generate_for_one_z
    dir_to_save_z_realizations=args.dir_to_save_z_realizations    
    num_smoothing_conv_layers = args.num_smoothing_conv_layers
    last_layer_size=args.last_layer_size
    n_days=args.n_days
    label_smoothing=args.label_smoothing
    train_G_with_context = args.train_G_with_context
    train_D_with_lowres_context = args.train_D_with_lowres_context
    train_D_with_highres_context=args.train_D_with_highres_context
    start_lr = args.start_lr 
    end_lr = args.end_lr
    add_mean_loss = args.add_mean_loss
    partition_start = args.partition_start


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
        save_model_dir=save_model_dir,
        z_shape=z_shape,
        pretrained_model=pretrained_model,
        n_generate_for_one_z=n_generate_for_one_z,
        dir_to_save_z_realizations=dir_to_save_z_realizations,
        num_smoothing_conv_layers=num_smoothing_conv_layers,
        last_layer_size=last_layer_size, 
        n_days=n_days,
        label_smoothing=label_smoothing,
        train_G_with_context=train_G_with_context,
        train_D_with_lowres_context=train_D_with_lowres_context,
        train_D_with_highres_context=train_D_with_highres_context,
        start_lr=start_lr,
        end_lr=end_lr,
        add_mean_loss=add_mean_loss,
        partition_start=partition_start)
    
    ex.run()
