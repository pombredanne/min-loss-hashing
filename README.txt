This is an implementation of the algorithm peresented in the paper:
"Minimal Loss Hashing for Compact Binary Codes", Mohammad Norouzi,
David Fleet, ICML 2010.


res/ contains codes and W matrices for semantic 22K Labelme dataset.
It will store the results for other datasets too.


data/ contains dataset files -- should be downloaded separately from

1- mtx files for creating the small databases (MNIST, LabelMe,
Peekaboom, Photo-Tourism, Nuresery) could be downloaded from
http://www.eecs.berkeley.edu/~kulis/data/

2- mat  files of the  pre-created data structures for  small databases
with    train/test    splits   is    available    for   download    at
http://www.cs.toronto.edu/~norouzi/get/small-datasets.tar

3- data/LabelMe_gist.mat is the 22K labelme database availabe at
http://cs.nyu.edu/~fergus/research/tfw_cvpr08_code.zip


RUN.m: is the starting point.  It shows how the code has been run for
small datasets and euclidean 22K LabelMe. It also includes the codes
for creating the plots.


MLH.m: is the function that calls learnMLH useful for validation over
a number of parameters.


learnMLH.m: is the main file for learning hash functions.


create_data: a function that creates dataset structures from different
sources of data based on input parameters.


create_training: performs train/validation/test splits


utils/ includes small functions that are used throughout the
code. Some of the functions are adapted from Spectral Hashing (SH),
Small Codes with Large Databases, Binary Reconstructive Embedding
(BRE) codes, generously provided by their authors.


plots/ contains some functions useful for plotting the curves in the
paper.
