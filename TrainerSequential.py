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
from torch.autograd import Variable
from DataSampler import DataSampler
import logging
from sacred import Experiment
from sacred.observers import MongoObserver
import Utils as ut
from Utils import GaussianNoise
from Utils import sort_files_by_size, snake_data_partition
#import torch.distributed as dist
import argparse
import logging
import csv

exp_id = 41
ex = Experiment('Exp 41: Pre-Training GAN as Autoencoder  saving model after 500 updates, z = 256')

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
    #z_shape = 256
    n_days = 32
    apply_norm = True
    #hyperparamteres
    label_smoothing = False
    add_noise = False
    experience_replay = False
    replay_buffer_size = batch_size * 20
    lr = 0.0002 #NOTE: this lr is coming from the original paper about DCGAN
    l1_lambda = 10
	
class TrainerSequential:
    @ex.capture
    def __init__(self):
        self.lon, self.lat, self.context_length, self.channels,  self.n_days, self.apply_norm, self.data_dir = self.set_parameters()
        self.label_smoothing, self.add_noise, self.experience_replay, self.batch_size, self.lr, self.l1_lambda, self.num_epoch, self.replay_buffer_size, self.report_avg_loss, self.gen_data_dir, self.real_data_dir, self.save_gen_data_update, self.n_data_to_save, self.norms_dir, self.pretrain, self.save_model_dir, self.is_autoencoder, self.z_shape  = self.set_hyperparameters()
        self.exp_id, self.exp_name, self._run = self.get_exp_info()
        #buffer for expereince replay        
        self.replay_buffer = []
        self.sorted_files = sort_files_by_size(self.data_dir)
        self.partition = snake_data_partition(self.sorted_files, 1) # WARNING: don't hardcode this here
        self.save_model = self.save_model_dir + 'exp_' + str(exp_id) + '/'


    @ex.capture
    def set_parameters(self, lon, lat, context_length, channels, n_days, apply_norm, data_dir):
        return lon, lat, context_length, channels,n_days, apply_norm, data_dir
    
    @ex.capture
    def set_hyperparameters(self, label_smoothing, add_noise, experience_replay, batch_size, lr, l1_lambda, num_epoch, replay_buffer_size, report_avg_loss, gen_data_dir, real_data_dir, save_gen_data_update, n_data_to_save, norms_dir, pretrain, save_model_dir, is_autoencoder, z_shape):
        return label_smoothing, add_noise, experience_replay, batch_size, lr, l1_lambda, num_epoch, replay_buffer_size, report_avg_loss, gen_data_dir, real_data_dir, save_gen_data_update, n_data_to_save, norms_dir, pretrain, save_model_dir, is_autoencoder, z_shape
    
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
        netD.apply(ut.weights_init)
        
        netG = Generator(self.lon, self.lat, self.context_length, self.channels, self.batch_size, self.z_shape)        
        
        if not self.pretrain:
            params = torch.load(self.save_model + "model.pt")
            netG.load_state_dict(params)
        else:
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
        
        logging.info("here")

        #reset the batch size based on the number of processes used
        self.total_batch_size  = self.batch_size        
        self.actual_batch_size = 2 # WARNING: DON'T HARDCODE THIS   #self.batch_size // comm_size
        self.num_workers       = self.batch_size // self.actual_batch_size


        self.replay_buffer_size = 20 * self.total_batch_size
        n_data_to_save_on_process = self.n_data_to_save #// comm_size #should be on CPU
        
        real_data_saved, gen_data_saved = [], []        

        rank = 0
        partition = self.partition[rank] #WARNING: testing using just 3 files from a dataset
        partition = partition[:len(partition)//2]#warning take just a half of a data
        logging.info("partition {}".format(partition))
        ds = NETCDFDataPartition(partition, self.data_dir)
        logging.info("finished loading partition")
        sampler = DataSampler(self.total_batch_size, ds.data, self.context_length, self.n_days)
        b_sampler = data.BatchSampler(sampler, batch_size=self.total_batch_size, drop_last=True)
        dl = data.DataLoader(ds, batch_sampler=b_sampler)
        dl_iter = iter(dl)
    
        loss_n_updates, report_step = 0, 0
        n_real_saved, n_gen_saved = 0, 0
        dates_mapping = []
        min_loss = float('inf')
            
        #Training loop
        for current_epoch in range(1, self.num_epoch + 1):        
            n_updates = 1
            logging.info("here")
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
        
                if self.label_smoothing:
                    ts = np.full((self.total_batch_size), 0.9)
                    real_labels = ut.to_variable(torch.FloatTensor(ts), requires_grad = False)
                else:
                    real_labels = ut.to_variable(torch.FloatTensor(np.ones(self.total_batch_size, dtype=int)),requires_grad = False)
                                    
                fake_labels = ut.to_variable(torch.FloatTensor(np.zeros(self.total_batch_size, dtype = int)), requires_grad = False)
                
                real_labels = real_labels.to(device)
                fake_labels = fake_labels.to(device)
                
                current_month = current_month.to(device) # NOTE: Maybe not yet
                avg_context = avg_context.to(device) # NOTE: Maybe not yet
                high_res_context = high_res_context.to(device) # NOTE: Maybe not yet

                #sample noise
                z = ds.get_noise(self.z_shape, self.batch_size)
                z = z.to(device)

                real_labelss = torch.chunk(real_labels,self.num_workers)
                fake_labelss = torch.chunk(fake_labels,self.num_workers)
                current_months = torch.chunk(current_month,self.num_workers)
                avg_contexts = torch.chunk(avg_context,self.num_workers)
                high_res_contexts = torch.chunk(high_res_context,self.num_workers)
                fake_outputs = []
                if self.is_autoencoder:
                    netG.zero_grad()
                    netD.zero_grad()
                    for worker in range(self.num_workers):
                        input = ds.build_input_for_D(current_months[worker], avg_contexts[worker], high_res_contexts[worker])            
                        outputs = netD(input).squeeze()
                        high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_contexts[worker], high_res_contexts[worker])
                        generated_month = netG(outputs, avg_ctxt_for_G, high_res_for_G)
                        reconstruction_loss = mse_loss_func(current_months[worker], generated_month)
                        logging.info("reconstruction_loss {} \n".format(reconstruction_loss))
                        logging.info("n_updates {} \n".format(n_updates))
                        self._run.log_scalar('reconstruct_loss', reconstruction_loss.item(), n_updates)
                        reconstruction_loss.backward()
                        fake_outputs.append(generated_month.to(cpu_dev))
                    fake_outputs = torch.cat(fake_outputs, 0)
                    g_optim.step()
                    d_optim.step()

                if self.pretrain and not self.is_autoencoder:
                    netG.zero_grad()
                    for worker in range(self.num_workers):
                        high_res_for_G, avg_ctxt_for_G = ds.reshape_context_for_G(avg_contexts[worker], high_res_contexts[worker])
                        fake_outputs = netG(z, avg_ctxt_for_G, high_res_for_G)    
                        losses = pretrain_G(fake_outputs, current_months[worker], mse_loss_func, 1, device) # WARNING: comm_size hardcoded to 1
                        log_losses(self._run, losses, n_updates)
                        loss = losses["total_loss"]
                        total_loss = loss[0]
                        logging.info("epoch {}, rank {}, update {}, g loss {:0.18f}".format(current_epoch, 0, n_updates, total_loss.item())) # WARNING: rank hardcoded to 0
                        total_loss.backward()
                        #divide by number of iterations
                    g_optim.step()
                    
                if self.pretrain or self.is_autoencoder:
                    if n_updates == self.save_gen_data_update:
                        save_model(netG, self.save_model)
                    
                    
                    #save real data to calculate statistics
                    if n_updates >= self.save_gen_data_update and n_real_saved < n_data_to_save_on_process:
                        current_month_to_save = current_month.to(cpu_dev)
                        real_data_saved.append(current_month_to_save)
                        n_real_saved += self.total_batch_size
                        dates_mapping.append(year_month_date)
                        if n_real_saved >= n_data_to_save_on_process:
                            save_data(real_data_saved, self.real_data_dir, rank, exp_id, dates_mapping)
                            #free buffer
                            real_data_saved = []
                            dates_mapping = []            

                    
                    #save fake data to calculate statistics
                    if n_updates >= self.save_gen_data_update and n_gen_saved < n_data_to_save_on_process:
                        fake_outputs = fake_outputs.to(cpu_dev)
                        gen_data_saved.append(fake_outputs)
                        n_gen_saved += self.total_batch_size
                        if n_gen_saved >= n_data_to_save_on_process:
                            save_data(gen_data_saved, self.gen_data_dir, rank, exp_id, dates_mapping)
                            #free buffer
                            gen_data_saved = []
                
                n_updates += 1    


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


def pretrain_G(fake_outputs, current_month_batch, mse_loss_func, comm_size, device):
    #compute targets
    map_mean_target = get_mean_map(current_month_batch, comm_size)
    map_std_target = get_std_map(current_month_batch, comm_size)
    
    #compute stat for fake
    batch_fake_outputs_mean = get_mean_map_for_fake(fake_outputs)
    batch_fake_outputs_std = get_std_map_for_fake(fake_outputs)
    
    #compute losses for means for each channel
    loss_mean_0 = mse_loss_func(map_mean_target[0], batch_fake_outputs_mean[0])
    loss_mean_1 = mse_loss_func(map_mean_target[1], batch_fake_outputs_mean[1])
    loss_mean_2 = mse_loss_func(map_mean_target[2], batch_fake_outputs_mean[2])
    loss_mean_3 = mse_loss_func(map_mean_target[3], batch_fake_outputs_mean[3])
    loss_mean_4 = mse_loss_func(map_mean_target[4], batch_fake_outputs_mean[4])
    loss_mean_5 = mse_loss_func(map_mean_target[5], batch_fake_outputs_mean[5])
    loss_mean_6 = mse_loss_func(map_mean_target[6], batch_fake_outputs_mean[6])
    
    total_loss = loss_mean_0 + loss_mean_1 + loss_mean_2 + loss_mean_3 + loss_mean_4 + loss_mean_5 + loss_mean_6

    #compute losses for stds for each channel
    loss_std_0 = mse_loss_func(map_std_target[0], batch_fake_outputs_std[0])
    loss_std_1 = mse_loss_func(map_std_target[1], batch_fake_outputs_std[1])
    loss_std_2 = mse_loss_func(map_std_target[2], batch_fake_outputs_std[2])
    loss_std_3 = mse_loss_func(map_std_target[3], batch_fake_outputs_std[3])
    loss_std_4 = mse_loss_func(map_std_target[4], batch_fake_outputs_std[4])
    loss_std_5 = mse_loss_func(map_std_target[5], batch_fake_outputs_std[5])
    loss_std_6 = mse_loss_func(map_std_target[6], batch_fake_outputs_std[6])

    total_loss += loss_std_0 + loss_std_1 + loss_std_2 + loss_std_3 + loss_std_4 + loss_std_5 + loss_std_6

    tas_loss, rhs_loss = reg_rh_and_tas(fake_outputs, device)
    total_loss += tas_loss + rhs_loss
        
    loss = {
        "mean_losses" : [loss_mean_0, loss_mean_1, loss_mean_2, loss_mean_3, loss_mean_4, loss_mean_5, loss_mean_6],
        "std_losses" : [loss_std_0, loss_std_1, loss_std_2, loss_std_3, loss_std_4, loss_std_5, loss_std_6],
        "rhs_loss" : [rhs_loss],
        "tas_loss" : [tas_loss],
        "total_loss" : [total_loss]
    }
    
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
    
def get_mean_map_for_fake(batch):
    bsz = batch.shape[0]
    N = 32 * bsz
    tsr = batch.permute(1,2,3,0,4).contiguous()
    tsr = tsr.view(n_channels, 128, 256, N)
    sum_tsr = tsr.sum(-1)
    mean_tsr = sum_tsr / N
    return mean_tsr


def get_std_map_for_fake(batch):
    bsz = batch.shape[0]
    N = 32 * bsz
    tsr = batch.permute(1,2,3,0,4).contiguous()
    tsr = tsr.view(n_channels, 128, 256, N)
    sum_tsr = tsr.sum(-1)
    mean_tsr = sum_tsr / N
    mean_tsr.unsqueeze_(-1)#7 x 128 x 256 x 1
    mean_tsr = mean_tsr.expand(7, 128, 256, N)

    #calc sq diff on current batch on the process
    sq_diff = ((tsr - mean_tsr) ** 2).sum(-1)
    std_tsr = torch.sqrt(sq_diff / N)
    return std_tsr    


def get_std_map(batch, comm_size):
    bsz = batch.shape[0]
    N = 32 * bsz
    tsr = batch.permute(1,2,3,0,4).contiguous()
    tsr = tsr.view(n_channels, 128, 256, N) #7 x 128 x 256 x bsz*32
    
    #sum along days in the batch on the current process
    sum_tsr = tsr.sum(-1)
    #summ along days in the batch on all the processes
    all_reduce_sum(sum_tsr)
    mean_tsr = sum_tsr / (N * comm_size)
    mean_tsr.unsqueeze_(-1)#7 x 128 x 256 x 1
    mean_tsr = mean_tsr.expand(7, 128, 256, N) #7 x 128 x 256 x 64
    #calc sq diff on current batch on the process and sum them up along days dimension
    sq_diff = ((tsr - mean_tsr) ** 2).sum(-1)
    #calc sq diffs for all the tensors
    all_reduce_sum(sq_diff)
    std_tsr = torch.sqrt(sq_diff / (N * comm_size))
    return std_tsr

def reg_rh_and_tas(batch, device):
    bsz = batch.shape[0]
    N = 32 * bsz
    tsr = batch.permute(1,2,3,0,4).contiguous()
    tsr = tsr.view(n_channels, 128, 256, N)
    
    keys = list(clmt_vars.keys())
    tas = tsr[keys.index('tas')]
    tasmin = tsr[keys.index('tasmin')]
    tasmax = tsr[keys.index('tasmax')]

    rhs = tsr[keys.index('rhs')]
    rhsmin = tsr[keys.index('rhsmin')]
    rhsmax = tsr[keys.index('rhsmax')]

    #filter tasmin and tasmax
    tas_diff_1 = tas - tasmin
    notinrange_1 = tas_diff_1[tas_diff_1 < 0]
    tasmin_sq_sum = torch.sum((notinrange_1 ** 2))
    
    tas_diff_2 = tas - tasmax
    notinrange_2 = tas_diff_2[tas_diff_2 > 0]
    tasmax_sq_sum = torch.sum((notinrange_2 ** 2))
    
    #all_reduce_sum(tasmin_sq_sum)
    #all_reduce_sum(tasmax_sq_sum)
    
    tas_loss = (tasmin_sq_sum + tasmax_sq_sum) / (N * 128 * 256)    
    
    #filter rhsmin and rhsmax
    rhsdiff = rhs - rhsmin
    notinrange_rhs_1 = rhsdiff[rhsdiff < 0]
    rhsmin_sq_sum = torch.sum((notinrange_rhs_1 ** 2))
    rhsdiff = rhs - rhsmax
    notinrange_rhs_2 = rhsdiff[rhsdiff > 0]
    rhsmax_sq_sum = torch.sum((notinrange_rhs_2 ** 2))
    rhs_loss = (rhsmin_sq_sum + rhsmax_sq_sum) / (N * 128 * 256)
    #logging.info("tas_loss {}, rhs_loss {}".format(tas_loss, rhs_loss))
    return tas_loss, rhs_loss


    
def save_model(netG, dir):
    logging.info("saving the model")
    torch.save(netG.state_dict(), dir + 'model.pt')
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
    t = TrainerSequential()
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
    parser.add_argument('--is_autoencoder', type=int)
    parser.add_argument('--z_shape', type=int)

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
    pretrain=args.pretrain
    save_model_dir = args.save_model_dir
    is_autoencoder = args.is_autoencoder
    z_shape = args.z_shape
    
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
        save_model_dir=save_model_dir,
        is_autoencoder=is_autoencoder,
        z_shape=z_shape)
    
    ex.run()
