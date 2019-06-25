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

#for current_epoch in tqdm(range(1, num_epoch+1)):
#	n_updates = 1
#	for batch_idx in range(num_batches):
		
