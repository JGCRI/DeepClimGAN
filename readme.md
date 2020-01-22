# DeepClimGAN : Generative Adversarial Networks for climate generation

## Input Data

Input data for model development can be found on the PNNL cluster in
the `/pic/projects/GCAM/DeepClimGAN-input` directory.


## Files

#Constants.py
Contains some constants used, e.g. climate variables

#DataPreprocessor.py
Used to build PyTorch tensors out of the MIROC5 data.

#DataSampler.py
Iterator

#Discriminator.py
A part of a GAN architecture.

#Generator.py
A part of a GAN architecture.

#Generator_alt.py
Alternative variant to Generator. The main difference is that it has different algorithm of applying upconvolutions, so that the generated data does not have checkerbpard artifacts.

#NETCDFDataPartition.py
Samples datapoints from the PyTorch tensors. Also contains methods to reshape datapoints to be fed into Generator and Discriminator

#Normalizer.py
Used to normalize raw PyTorch tensors

#Trainer.py
Distributed training of a model (either 8 nodes, 2GPUs on each OR 2 nodes 8GPUs on each)

#TrainerAdversar.py
Adversarial training of a model. Sequential training, that means it uses just one node, but processes a batch in a sequence.

#TrainerSequential.py
Pre-training of a model. Sequential training, that means it uses just one node, but processes a batch in a sequence.

#Utils.py
Contains multiple utility methods

#get_statistics.py
Used as an initial evaluation method. Contains multiple methods to analyze the generated data.

#normalize_distrib.py
Distributed normalization of PyTorch tensor. Used together with Normalizer.py

#ops.py
Contains short forms of basic network blocks.

#plot_time_series.py
I think it is a redundtant file that is not used anymore..
