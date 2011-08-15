Minimal loss hashing for learning similarity preserving binary hash
functions. Copyright (c) 2011, Mohammad Norouzi <mohammad.n@gmail.com>
and David Fleet <fleet@cs.toronto.edu>. This is a free software; for
license information please refer to license.txt file.

~~~~~~~~~~~~~ Reference

This is an implementation of the algorithm presented in the paper
"Minimal Loss Hashing for Compact Binary Codes, Mohammad Norouzi,
David J Fleet, ICML 2010", with slight modifications.


~~~~~~~~~~~~~ Mex compilation (optional)

Compile utils/hammingDist2.cpp, which is a faster alternative to
hammingDist.m. The mex implementation uses GCC built-in popcount
function. Please make changes to eval_linear_hash.m and eval_labelme.m
to use hammingDist2.

~~~~~~~~~~~~~ Usage

download dataset files (see below)
RUN.m includes code snippets for different experiments

~~~~~~~~~~~~~ List of files

data/ folder will contain dataset files which you should download
separately:

- data/LabelMe_gist.mat is the 22K LabelMe dataset available from
http://cs.nyu.edu/~fergus/research/tfw_cvpr08_code.zip, courtesy of
Rob Fergus.

- data/kulis/*.mtx files for 5 small datasets (MNIST, LabelMe,
Peekaboom, Photo-Tourism, Nursery) can be downloaded from
http://www.eecs.berkeley.edu/~kulis/data/, courtesy of Brian
Kulis. (store the files under data/kulis/ sub-directory)

RUN.m: is the starting point.  It shows how the code can be run for
Euclidean/Semantic 22K LabelMe and small datasets. It also includes
the codes for creating the figures.

learnMLH.m: the main file for learning hash functions. It performs
stochastic gradient descent to learn the hash parameters.

MLH.m: performs validation on sets of parameters by calling
appropriate instances of learnMLH function.

create_data: a function that creates dataset structures from different
sources of data based on its input parameters.

create_training: performs train/validation/test splits

utils/ folder includes small functions that are used throughout the
code. Some of the functions are adapted from Spectral Hashing (SH)
source code generously provided by Y. Weiss, A. Torralba, R. Fergus.

plots/ folder contains some functions useful for plotting the curves
used in the paper.

res/ folder will store the result files. Pre-trained parameter
matrices and binary codes for semantic 22K LabelMe are already there.

... (incomplete)

~~~~~~~~~~~~~ Notes

This implementation is slightly different from the algorithm presented
in the MLH ICML'11 paper. Main modifications include 1) an L2
regularizer on W matrix is used instead of fixing the norm of W. Thus
instead of tuning epsilon parameter which gets multiplied by the loss
function, we tune a regularizer parameter and do not change loss. 2)
For balancing precision and recall, instead of formulating a parameter
lambda inside the hinge loss, we re-define lambda as the ratio of
positive and negative pairs to be sampled during training. We usually
use lambda=.5 meaning equal sampling of positive and negative
pairs. For one of the experiments we set lambda=0 meaning the original
distribution of positive and negative pairs.
