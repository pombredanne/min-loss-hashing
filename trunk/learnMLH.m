function [best_ap best_W Wall best_params] = learnMLH(data, param, verbose, initW)

Ntraining = data.Ntraining;
NtrainingSqr = Ntraining^2;
Xtraining = data.Xtraining;
dtr = size(Xtraining, 2);
nb = param.nb;
initeta = param.eta;
shrink_eta = param.shrink_eta;
ratio_loss_pos = param.ratio_loss_pos;
ratio_loss_neg = param.ratio_loss_neg;
size_batches = param.size_batches;
maxiter = param.maxiter;
rho = param.rho;
momentum = param.momentum;
zerobias = param.zerobias;
if strcmp(param.losstype, 'BRE')
  % Normalizing distance
  data.Dtraining = data.Dtraining / max(data.Dtraining(:));
end

if (verbose)
  fprintf('---------------------------\n');
  fprintf('nb = %d\n', nb);
  fprintf('init eta = %.4f\n', initeta);
  fprintf('rloss+ = %.3f\n', ratio_loss_pos);
  fprintf('rloss- = %.3f\n', ratio_loss_neg);
  fprintf('size mini-batches = %d\n', size_batches);
  fprintf('max iter = %d\n', maxiter);
  fprintf('rho = %d\n', rho);
  fprintf('zerobias = %d\n', zerobias);
  fprintf('verbose = %d\n', verbose);
  fprintf('doval = %d\n', param.doval);
  fprintf('trainset = ''%s''\n', param.trainset);
  fprintf('losstype = ''%s''\n', param.losstype);
  fprintf('weight decay = ''%.5f''\n', param.shrink_w(1));
  fprintf('shrink eta (0/1) = ''%d''\n', param.shrink_eta);
  fprintf('momentum = ''%.2f''\n', momentum);
  fprintf('---------------------------\n');
end

% initW is passed from ouside
if (exist('initW'))
  W = initW;
else
  W = [randn(nb, dtr) zeros(nb,1)]; % LSH
end

W = W./[repmat(sqrt(sum(W(:,1:end-1).^2,2)), [1 dtr+1])]; % Normalizing W
Winit = W;
Wall{1} = Winit;
err = 0;

if (verbose && param.doval)
  eval(-1, initeta, data, W, rho);
end

% initialization
best_ap = -Inf;
avg_err = 0;
best_W = [];
best_params = param;
ncases = size_batches;
nnz = 0;
nnz_fp = 0; nnz_fn = 0; nnz_tp = 0; nnz_tn = 0;
FP = 0; TP = 0; FN = 0; TN = 0;
sum_resp = 0;
n_sum_resp = 0;
Winc = 0;
Winc_new = 0;

cases = ceil(rand(ncases, 1)*NtrainingSqr);
Cases = ceil(rand(10^5, 1)*NtrainingSqr);
for t=1:maxiter+5*(param.doval>0)
  fprintf('%d\r', t);
  if (shrink_eta)
    eta = initeta/10 + 9/10*initeta * (1 + maxiter+[(param.doval>0)*5] - t)/(maxiter+[(param.doval>0)*5]);
  else
    eta = initeta;
  end
  
  if (verbose == 2 & t <= maxiter) % Check whether the upper-bound gets decreased.
    ncases = 10^5;
    ncases2 = 10^5;
    [x1 x2] = ind2sub([Ntraining Ntraining], Cases);
    l = data.WtrueTraining(Cases)';  
     
    x1 = [Xtraining(x1(:), :) ones(ncases,1)]';
    x2 = [Xtraining(x2(:), :) ones(ncases,1)]';
    
    Wx1 = W*x1;
    Wx2 = W*x2;
    y1 = sign(Wx1);
    y2 = sign(Wx2);
  
    y1plus = Wx1;
    y1minus = -Wx1;
    y2plus = Wx2;
    y2minus = -Wx2;
    [valeq indeq]   = max([cat(3, y1plus+y2plus, y1minus+y2minus)],[],3);
    [valneq indneq] = max([cat(3, y1plus+y2minus, y1minus+y2plus)],[],3);
    % valeq and valneq are of the size nb*ncases.
    [val ind] = sort(valneq-valeq, 'descend');

    if (strcmp(param.losstype, 'hinge'))
      % creating the hinge-like loss for the positive and negative pairs
      loss = ratio_loss_pos * [kron(l == 1, [zeros(1, rho) 1:(nb-rho+1)]')] ...
      	     + ratio_loss_neg * [kron(l == 0, [(rho+1):-1:1 zeros(1, nb-rho)]')];
      % loss = ratio_loss_pos * [kron(l == 1, ([zeros(1, rho) 1:(nb-rho+1)]/(nb-rho+1))')] ...
      % 	     + ratio_loss_neg * [kron(l == 0, ([(rho+1):-1:1 zeros(1, nb-rho)]/(rho+1))')];
    elseif (strcmp(param.losstype, 'BRE'))
      % creating the quadratic loss function of the BRE method
      d = data.Dtraining(Cases);
      loss = ratio_loss_pos * [(repmat(d, [1 nb+1])-repmat([0:nb]/nb, [ncases 1])).^2];
      loss = loss';
    else
      error('losstype is not supported.\n');
    end
        
    [v nflip] = max([zeros(1, ncases2); cumsum(val)] + loss);  % determining number of flips needed for loss-adjusted inference
    nflip = nflip - 1;
    % Deriving the individual codes that solve loss-adjusted inference
    y1p = zeros(size(y1));
    y2p = zeros(size(y2));    
    tmp = repmat((1:nb)',[1 ncases2]) <= repmat(nflip, [nb 1]);
    ind = squeeze(ind) + repmat(0:nb:(ncases2-1)*nb,[nb, 1]);
    ll = ind(tmp);
  
    y1p(ll) = 2*(2-indneq(ll))-1;
    y2p(ll) = 2*(indneq(ll)-1)-1;
    notll = find(y1p == 0);
    y1p(notll) = 2*(2-indeq(notll))-1;
    y2p(notll) = 2*(2-indeq(notll))-1;

    % Computing the value of upper-bound
    tmp = [x1*(-y1'+y1p') + x2*(-y2'+y2p')]';
    ind_offset = repmat(nb+1,[ncases2,1]).*(0:ncases2-1)';
    
    upperbound = sum(sum(W .* tmp)) + sum(loss(nflip(:)+1 +ind_offset));
    fprintf('%.2f\t', upperbound); 
    hdis = full(nb-(nb/2+(sum(y1.*y2)/2)));
    act_loss = sum(loss(hdis(:)+1 +ind_offset));
    fprintf('%.2f\n', full(act_loss)); 
  end

  ntraining = 10^5; % min(10^5, NtrainingSqr);
  ncases = size_batches;
  nbatches = floor(ntraining / ncases);
  for b=1:nbatches
% we don't re-normalize W anymore, instead we use weight decay i.e., L2 norm regularizer
%    normW = [repmat(sqrt(sum(W(:,1:end-1).^2,2)), [1 dtr+1])];
%    W = W./normW;
    cases = ceil(rand(ncases, 1)*NtrainingSqr);
    [x1 x2] = ind2sub([Ntraining Ntraining], cases);
    l = data.WtrueTraining(cases)';
      
    x1 = [Xtraining(x1(:), :) ones(ncases,1)]';
    x2 = [Xtraining(x2(:), :) ones(ncases,1)]';
    Wx1 = W*x1;
    Wx2 = W*x2;
    y1 = sign(Wx1);
    y2 = sign(Wx2);

    sum_resp = sum_resp + sum(y1 == 1, 2);
    n_sum_resp = n_sum_resp + size(y1, 2);
    r = sum_resp / n_sum_resp;
    ind_sm_ratio = (min(r, 1 - r)) < .03;
    n_sm_ratio = sum(ind_sm_ratio);
    
    hdis = nb-(nb/2+(sum(y1.*y2)/2));
    TP = TP + sum((hdis <= rho) & (l == 1));
    TN = TN + sum((hdis >  rho) & (l == 0));
    FP = FP + sum((hdis <= rho) & (l == 0));
    FN = FN + sum((hdis >  rho) & (l == 1));

    y1plus = Wx1;
    y1minus = -Wx1;
    y2plus = Wx2;
    y2minus = -Wx2;
       
    [valeq indeq]   = max([cat(3, y1plus+y2plus, y1minus+y2minus)],[],3);
    [valneq indneq] = max([cat(3, y1plus+y2minus, y1minus+y2plus)],[],3);
    % valeq and valneq are of the size nb*ncases.
    [val ind] = sort(valneq-valeq, 'descend');

    if (strcmp(param.losstype, 'hinge'))
      % creating the loss for the positive and negative pairs
      loss = ratio_loss_pos * [kron(l == 1, ([zeros(1, rho) 1:(nb-rho+1)]/(nb-rho+1))')] ...
	     + ratio_loss_neg * [kron(l == 0, ([(rho+1):-1:1 zeros(1, nb-rho)]/(rho+1))')];
    elseif (strcmp(param.losstype, 'BRE'))
      % creating the loss based on the BRE method
      d = data.Dtraining(cases);
      loss = ratio_loss_pos * [(repmat(d, [1 nb+1])-repmat([0:nb]/nb, [ncases 1])).^2];
      loss = loss';
    else
      error('losstype is not supported.\n');
    end

    [v nflip] = max([zeros(1, ncases); cumsum(val)] + loss);
    nflip = nflip - 1;
    y1p = zeros(size(y1));
    y2p = zeros(size(y2));    
    tmp = repmat((1:nb)',[1 ncases]) <= repmat(nflip, [nb 1]);
    ind = squeeze(ind) + repmat(0:nb:(ncases-1)*nb,[nb, 1]);
    ll = ind(tmp);

    y1p(ll) = 2*(2-indneq(ll))-1;
    y2p(ll) = 2*(indneq(ll)-1)-1; 
    notll = find(y1p == 0);
    y1p(notll) = 2*(2-indeq(notll))-1;
    y2p(notll) = 2*(2-indeq(notll))-1;
    
    nonzero_grad = sum(abs(y1-y1p) + abs(y2-y2p)) ~= 0;
    nnz = nnz + sum(nonzero_grad);
    nnz_fn = nnz_fn + sum(nonzero_grad & (hdis >  rho) & (l == 1));
    nnz_fp = nnz_fp + sum(nonzero_grad & (hdis <= rho) & (l == 0));
    nnz_tp = nnz_tp + sum(nonzero_grad & (hdis <= rho) & (l == 1));
    nnz_tn = nnz_tn + sum(nonzero_grad & (hdis >  rho) & (l == 0));

    tmp = [x1(:,nonzero_grad)*(y1(:,nonzero_grad)-y1p(:,nonzero_grad))' + x2(:,nonzero_grad)*(y2(:,nonzero_grad)-y2p(:,nonzero_grad))']';
    batcherr =  sum(sum(W .* -tmp)) + sum(loss(nflip(:)+1+repmat(nb+1,[ncases,1]).*(0:ncases-1)'));
    err = err + batcherr;
    Winc_new = tmp - ncases * param.shrink_w * W;
        
    Winc = momentum * Winc + eta * Winc_new / ncases;
    
    if (zerobias)
      W(:, 1:end-1) = W(:, 1:end-1) + Winc(:, 1:end-1);
    else
      W = W + Winc;
    end
        
    nnz = 0;
  end
  
  if (verbose && mod(t, floor(maxiter/verbose)) == 0 && t < maxiter)
    fprintf(['    ~~err:%.3f---nnz:%.1f---FN[%.0f%% %d]---FP[%.0f%% %d]---TP[%.0f%% %d]---TN[%.0f%% %d]' ...
	     ' prec:%.3f | rec:%.3f | ||W||=%.2f | nsm:%d (%.3f)'], err, nnz/nbatches, 100*nnz_fn/FN, FN, 100*nnz_fp/FP, ...
	    FP, 100*nnz_tp/TP, TP, 100*nnz_tn/TN, TN, 100*TP/(TP+FP), TP/(TP+FN), sqrt(sum(W{end}(:).^2)), ...
	    n_sm_ratio, min(min(r, 1 - r)));
    if (verbose)
      fprintf('\n');
    else
      fprintf('\r');
    end;
    
    err = 0;
    
    nnz_fp = 0; nnz_fn = 0; nnz_tp = 0; nnz_tn = 0;
    FP = 0; TP = 0; FN = 0; TN = 0;
    sum_resp = 0;
    n_sum_resp = 0;
  
    Wall{t+1} = W;
  end
  nnz = 0;

  if(param.doval)
    if (mod(t, floor(maxiter/(param.doval))) == 0 && t < maxiter)
      eval(t, eta, data, W, rho);
    end
  end
  
  if(param.doval && t >= maxiter)
    [ap err] = eval(t, eta, data, W, rho);
    avg_err = avg_err + err;
    if isinf(best_ap)
      n_ap = 1;
      best_ap = ap;
    else
      n_ap = n_ap + 1;
      best_ap = best_ap + ap;
    end
  end
end

if (exist('n_ap'))
  best_ap = best_ap / n_ap;
  fprintf('The mean AP over last %d steps: %.3f                         \n', n_ap, best_ap);
  if (strcmp(data.MODE, 'sem-full-mnist')) %mnist
    fprintf('The mean error over last %d steps: %.3f                         \n', n_ap, avg_err/n_ap);
  end 
end
best_param.ap = best_ap;

best_W = W;
best_params = param;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ap err] = eval(t, eta, data, W, rho)

err = NaN;
if (strcmp(data.MODE, 'sem-full-mnist')) %mnist
  err = eval_mnist(W, data);
end

fprintf('(%d/%.3f) ', t, eta);
if (~isnan(err))
  fprintf('~~~ err: %.3f ', err);
end

[p1 r1] = eval_linear_hash(W, data);
p1(isnan(p1)) = 1;
if (strcmp(data.MODE, 'sem-22k-labelme'))
  % semantic labelme
  [ap pcode] = eval_labelme(W, data);
  fprintf('~~~ prec: %.3f ~~~ recall: %.4f ~~~ ap: %.5f ~~~ ap(50): %.3f ~~~ ap(100): %.3f ~~~ ap(500): %.3f\n', ...
	  p1(rho+1), r1(rho+1), ap, pcode(50), pcode(100), pcode(500));
else
  ap = sum([(p1(1:end-1)+p1(2:end))/2].*[(r1(2:end)-r1(1:end-1))]);
  fprintf('~~~ prec: %.3f ~~~ recall: %.4f ~~~ ap: %.5f\n',  p1(rho+1), r1(rho+1), ap);
end
