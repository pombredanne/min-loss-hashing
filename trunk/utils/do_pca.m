function data = do_pca(data, nb);
  % performing PCA on the data structure
  npca = min(nb, size(data.Xtraining, 2));
  opts.disp = 0;
  [pc, l] = eigs(cov(data.Xtraining), npca, 'LM', opts);
  data.Xtraining = data.Xtraining*pc;
  if (isfield(data, 'Xtest'))
    data.Xtest = data.Xtest*pc;
  end

  data.princComp = pc;
