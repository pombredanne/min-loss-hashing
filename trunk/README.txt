Minimal loss hashing code for learning similarity preserving binary
hash function. Copyright (C) 2011 Mohammad Norouzi and David Fleet.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

A copy of the GNU General Public License v3.0 can be found at
license.txt, see <http://www.gnu.org/licenses/>.


~~~~~~~~~~~~~ Reference

This is an implementation of the algorithm presented in the paper
"Minimal Loss Hashing for Compact Binary Codes", Mohammad Norouzi,
David J Fleet, ICML 2010.


~~~~~~~~~~~~~ Mex compilation (optional)

Compile utils/hammingDist2.cpp, which is a faster alternative to
hammingDist.m. The mex implementation uses GCC built-in popcount
function. Please make sure to change eval_linear_hash.m accordingly.

~~~~~~~~~~~~~ Usage

see RUN.m

~~~~~~~~~~~~~ List of files

data/ folder contains dataset files - should be downloaded separately
from:

- data/kulis/*.mtx files for creating the small databases (MNIST,
LabelMe, Peekaboom, Photo-Tourism, Nursery) could be downloaded from
http://www.eecs.berkeley.edu/~kulis/data/. should be stored under
data/kulis/

- data/LabelMe_gist.mat is the 22K labelme database available at
http://cs.nyu.edu/~fergus/research/tfw_cvpr08_code.zip

RUN.m: is the starting point.  It shows how the code can be run for
Euclidean 22K LabelMe and small datasets. It also includes the codes
for creating the figures.

learnMLH.m: is the main file for learning hash functions. Performs
stochastic gradient descent to learn hash parameters.

MLH.m: is the function that calls learnMLH useful for validation over
a number of parameters.

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
