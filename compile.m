mex loss_adj_inf_mex.cpp CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp";
% if you don't have -fopenmp option, just remove it,
% and you will lose multicore functionality, but that
% should be fine.

mex utils/hammDist_mex.cpp -outdir utils;
mex utils/accumarray_reverse.cpp -outdir utils;
