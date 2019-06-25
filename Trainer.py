
from DataLoader import DataLoader

#Default params

#Percent of data to use for training 
train_pct = 0.7
batch_size = 1



def main():

        print("Started parsing data...")

        d_loader = DataLoader()
        splitted_tsr = d_loader.build_dataset()

        print("Started splitting data..")

        train_x, dev_x, test_x = d_loader.split_dataset(splitted_tsr, train_pct)
	
        print("The data has been splitted to train, dev and test sets")


main()
