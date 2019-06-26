import numpy as np
import os
import sys
from DataLoader import DataLoader
from Discriminator import Discriminator
from Generator import Generator



def weights_init(m):
        """
        Custom weights initialization called on	Generator and Discriminator
        param: m ()

        return:	None
        """

        classname = m.__class__.name__
        if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

def to_variable(x, requires_grad = True):
        if torch.cuda.is_available():
                x = x.cuda()
        return Variable(x, requires_grad)


#Default params
data_dir = '../clmt_data/'

#Percent of data to use for training 
train_pct = 0.7
batch_size = 1

#dimension of Gaussian distribution to sample noise from
z_shape = 100



num_epoch = 5
batch_size = 64
lr = 0.0002
l1_lambda = 10

netD = Discriminator()
netG = Generator()
netD.apply(weights_init)
netG.appl(weights_init)

if torch.cuda.is_available():
	netD.cuda()
	netG.cuda()


loss_func = torch.nn.MSELoss()
d_optim = torch.optim.Adam(netD.parameters(), lr, [0.5, 0.999])
g_optim = torch.optim.Adam(netG.parameters(), lr, [0.5, 0.999])





print("Started parsing data...")
d__loader = DataLoader()
splitted_tsr = d_loader.build_dataset(data_dir)

print("Started splitting data..")

train_x, dev_x, test_x = d_loader.split_dataset(splitted_tsr, train_pct)
	
print("The data has been splitted to train, dev and test sets")
	
num_batches = train_x.shape[0] / batch_size

for current_epoch in tqdm(range(1, num_epoch+1)):
	n_updates = 1
	for batch_idx in range(num_batches):
		inputs_real = d_loader.get_batch(train_x, batch_size, shuffle=True)#TODO: check shuffle attribute
		real_labels = to_variable(torch.LongTensor(np.ones(batch_size, dtype = int)), requires_grad = False)
		fake_labels = to_variable(torch.LongTensor(np.zeros(batch_size, dtype = int)), requires_grad = False)
		

	if n_updates % 2 ==1:
		#train Discriminator
		netD.zero_grad()
		netG.zero_grad()
		outputs = netD(inputs_real).squeeze()
		d_real_loss = loss_func(outputs, real_labels)
		z = d_loader.get_noise(z_shape)
		fake_inputs = netG(z)
		outputs = netD(fake_inputs).squeeze()
		d_fake_loss = loss_func(outputs, fake_labels)
		d_loss = d_real_loss + d_fake_loss
		d_loss.backward()
		d_optim.step()
		
	else:
		#train Generator
		netD.zero_grad()
		netG.zero_grad()
		
		g_outputs_real = netG(inputs_real)#NOTE sure about this part
		outputs = netD(g_outputs_real).squeeze()
		reg_loss = torch.mean(torch.abs(inputs_real - g_outputs_real)) * l1_lambda
		g_loss = loss_func(outputs, real_labels) + reg_loss
		g_loss.backward()
		g_optim.step()
		
	


	n_updates += 1
					
