function [WW] = MLH(data_in, loss_cell, nb, eta_set, momentum, size_batches_set, trainset, maxiter, ...
					zerobias_set, doval_during, doval_after, verbose, shrink_w_set, shrink_eta)

initW = [.1*randn(nb, size(data_in.Xtraining, 1)) zeros(nb, 1)]; % LSH
best_ap = -Inf;
best_prec = -Inf;

data = create_training(data_in, trainset, doval_during + doval_after);
if (verbose)
  display(data);
end

losstype = loss_cell{1};
if strcmp(losstype, 'hinge')
  rho_set = loss_cell{2};
  lambda_set = loss_cell{3};
  m = 1;
  for rho = rho_set
    for lambda = lambda_set
      loss_set(m).type = losstype;
      loss_set(m).rho = rho;
      loss_set(m).lambda = lambda;
      m = m+1;
    end
  end 
else
  loss_set(1).type = losstype;
end


n = 1;
for size_batches = size_batches_set
for eta = eta_set
for shrink_w = shrink_w_set
for loss = loss_set
for zerobias = zerobias_set
  
  param.size_batches = size_batches;
  param.loss = loss;
  param.shrink_w = shrink_w;
  param.nb = nb;
  param.eta = eta;
  param.maxiter = maxiter;
  param.momentum = momentum;
  param.zerobias = zerobias;
  param.trainset = trainset;
  param.mode = data.MODE;
  param.Ntraining = data.Ntraining;
  param.doval_during = doval_during;
  param.doval_after = doval_after;
  param.shrink_eta = shrink_eta;
    
  [ap W Wall params] = learnMLH(data, param, verbose, initW);
  
  if (~verbose)
    if (numel(size_batches_set) > 1)
      fprintf('batch-size: %d  ', size_batches);
    end
    if (numel(loss_set) > 1)
      if strcmp(loss.type, 'hinge')
	fprintf('rho:%d / lambda:%d  ', loss.rho, loss.lambda);
      end
    end
    if (numel(eta_set) > 1)
      fprintf('eta: %.3f  ', eta);
    end
    if (numel(shrink_w_set) > 1)
      fprintf('shrink_w: %.6f  ', shrink_w);
    end
    fprintf(' --> ap:%.3f\n', ap);
  end
  
  WW(n).ap = ap;
  WW(n).W = W;
  WW(n).params = params;
  WW(n).mode = data_in.MODE;
  
  % Because PCA is not necessarily unique, we store the prinicipal components of the data with the
  % learned weights too.
  if (isfield(data_in, 'princComp'))
    WW(n).princComp = data_in.princComp;
  end
  n = n+1;

end
end
end
end
end