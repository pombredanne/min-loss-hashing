function [WW] = MLH(data_in, losstype, nb, eta_set, momentum, ratio_loss_set, ...
		    size_batches_set, trainset, maxiter, rho_set, zerobias, ...
		    doval, verbose, shrink_w_set, shrink_eta)

initW = [randn(nb, size(data_in.Xtraining, 2)) zeros(nb, 1)]; % LSH
best_ap = -Inf;
best_prec = -Inf;

data = create_training(data_in, trainset, doval);
if (verbose)
  display(data);
end

n = 1;
for size_batches = size_batches_set
for rho = rho_set
for iloss = 1:size(ratio_loss_set, 1)
for eta = eta_set
for shrink_w = shrink_w_set
     
  param.size_batches = size_batches;
  param.nb = nb;
  param.ratio_loss_pos = ratio_loss_set(iloss, 1);
  param.ratio_loss_neg = ratio_loss_set(iloss, 2);
  param.eta = eta;
  param.maxiter = maxiter;
  param.rho = rho;
  param.momentum = momentum;
  param.zerobias = zerobias;
  param.trainset = trainset;
  param.losstype = losstype;
  param.mode = data.MODE;
  param.Ntraining = data.Ntraining;
  param.shrink_w = shrink_w;
  param.doval = doval;
  param.shrink_eta = shrink_eta;
    
  [ap prec W Wall params] = learnMLH(data, param, verbose, initW);

  WW(n).ap = ap;
  WW(n).W = W;
  WW(n).params = params;
  WW(n).mode = data_in.MODE;
  n = n+1;

end
end
end
end
end
