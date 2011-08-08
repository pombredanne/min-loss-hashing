function [avg_ap best_W Wall best_params] = learnMLH(data, param, verbose, initW)

nb = param.nb;							% number of bits i.e, binary code length
initeta = param.eta;					% initial learning rate
shrink_eta = param.shrink_eta;			% whether shrink learning rate, as training proceeds
size_batches = param.size_batches;		% mini-batch size
maxiter = param.maxiter;				% number of gradient update iterations (each iteration
                                        % consists of 10^5 pairs)
zerobias = param.zerobias;				% whether offset terms are learned for hash hyper-planes
                                        % or they all go through the origin
momentum = param.momentum;				% momentum term (between 0 and 1) for gradient update
shrink_w = param.shrink_w;				% weight decay parameter

loss_func = param.loss;
% loss_function is a structure itself. The code supports loss_func.type = 'hinge' or 'bre'.  For
% hinge type loss function two other parameters 'loss_func.rho' and 'loss_func.lambda' are required.  For
% the bre loss no parameter is needed.

if strcmp(loss_func.type, 'bre')
  % Normalizing distance
  data.Dtraining = data.Dtraining / max(data.Dtraining(:));
end
Ntraining = data.Ntraining;
NtrainingSqr = Ntraining^2;
Xtraining = data.Xtraining;
indPosPairs = find(data.WtrueTraining);
if strcmp(loss_func.type, 'hinge')
  lambda = loss_func.lambda;
  rho = loss_func.rho;
else
  % rho is used to report precision and recall in eval(...) function below
  rho = ceil(2^(log(nb) / log(2) - 4) * 2);
end

if (verbose)
  fprintf('---------------------------\n');
  fprintf('nb = %d\n', nb);
  fprintf('losstype = ''%s''\n', loss_func.type);
  if strcmp(loss_func.type, 'hinge')
    fprintf('- loss rho = %d\n', rho);
    fprintf('- loss lambda = %.3f\n', lambda);
  end
  if (zerobias)
    fprintf('zero offset = yes\n');
  else
    fprintf('zero offset = no\n');
  end
  fprintf('weight decay = ''%.5f''\n', shrink_w);
  fprintf('trainset = ''%s''\n', param.trainset);
  fprintf('max iter = %d\n', maxiter);
  fprintf('init eta = %.4f\n', initeta);
  if (param.shrink_eta)
    fprintf('shrink eta = yes\n');
  else
    fprintf('shrink eta = no\n');
  end
  fprintf('size mini-batches = %d\n', size_batches);
  fprintf('momentum = ''%.2f''\n', momentum);
  fprintf('validation during / after training = %d / %d times\n', param.doval_during, param.doval_after);
  % more that one validation after training is suggested and averaging to account for validation noise
  fprintf('---------------------------\n');
end

if (exist('initW'))
  % initW is passed from ouside
  W = initW;
else
  % initialize W with LSH
  input_dim = size(Xtraining, 1);
  initW = [.1*randn(nb, input_dim) zeros(nb, 1)];
  % offset terms are initialized at zero
end

% Normalizing W
% W = W./[repmat(sqrt(sum(W(:,1:end-1).^2,2)), [1 size(W,2)])];
Wall{1} = W;
bound = 0;

if (verbose && param.doval_during)
  eval(-1, initeta, data, W, rho, verbose);
end
if (verbose)
  fprintf('norm W=%.2f\n', sqrt(sum(W(:).^2)));
end

% initialization
avg_ap  = 0;
avg_err = 0;
ncases = size_batches;
nnz = 0;
nnz_fp = 0; nnz_fn = 0; nnz_tp = 0; nnz_tn = 0;
FP = 0; TP = 0; FN = 0; TN = 0;
sum_resp = 0;
n_sum_resp = 0;
Winc = zeros(size(W));

ntraining = 10^5; 			% total number of pairs to be considered in each iteration
ncases = size_batches;
maxb = floor(ntraining / ncases);	% number of mini-batches
maxt = maxiter+param.doval_after-1;	% number of epochs

for t=1:maxt
  if (shrink_eta)			% learning rate update
    eta = initeta * (maxt-t)/(maxt);
  else
    eta = initeta;
  end

  n_bits_on    = 0;
  n_bits_total = 0;
  nnz = 0;
  bound = 0;

  for b=1:maxb
    % random selection of pairs
    cases = ceil(rand(ncases, 1)*NtrainingSqr);
    [x1nd x2nd] = ind2sub([Ntraining Ntraining], cases);
    
    x1 = Xtraining(:, x1nd(:));
    x2 = Xtraining(:, x2nd(:));
    
	l = full(data.WtrueTraining(cases)');
    
    x1 = [x1; ones(1,ncases)];
    Wx1 = W*x1;
    y1 = sign(Wx1);

    x2 = [x2; ones(1,ncases)];
    Wx2 = W*x2;
    y2 = sign(Wx2);			% we use -1/+1 instead of 0/1 values for the binary vectors

    y1plus  = Wx1;			% y1 bits all on
    y1minus = -Wx1;			% y1 bits all off
    y2plus  = Wx2;			% y2 bits all on
    y2minus = -Wx2;			% y2 bits all off

    [valeq indeq]   = max([cat(3, y1plus+y2plus, y1minus+y2minus)],[],3);
    % best score for bits in y1 and y2 being the same
    [valneq indneq] = max([cat(3, y1plus+y2minus, y1minus+y2plus)],[],3);
    % best score for bits in y1 and y2 being different
    % valeq and valneq are matrices of size nb*ncases.
    [val indval] = sort(valneq-valeq, 'descend');
        
    if (strcmp(loss_func.type, 'hinge'))
      % creating the hinge-like loss for the positive and negative pairs
       loss = (1/sqrt(lambda)) * [kron(l == 1, ([zeros(1, rho) 1:(nb-rho+1)] / (nb))')] ...
			  + sqrt(lambda)   * [kron(l == 0, ([(rho+1):-1:1 zeros(1, nb-rho)] / (nb))')];
    elseif (strcmp(loss_func.type, 'bre'))
      % creating the quadratic loss function of the BRE method
      % it requires the Dtraining matrix
      d = data.Dtraining(cases);
      loss = [(repmat(d, [1 nb+1])-repmat([0:nb]/nb, [ncases 1])).^2]';
    else
      error('losstype is not supported.\n');
    end

    [v nflip] = max([zeros(1, ncases); cumsum(val)] + loss); % loss-adjusted inference
    nflip = nflip - 1;			% number of different bits in the solution of loss-adjusted inference

    y1p = zeros(size(y1));		% (y1p, y2p) are the solutions to loss-adjusted inference
    y2p = zeros(size(y2));    
    tmp = repmat((1:nb)',[1 ncases]) <= repmat(nflip, [nb 1]);
    indval = squeeze(indval) + repmat(0:nb:(ncases-1)*nb,[nb, 1]); % working out the indices
    diffbits = indval(tmp);
    y1p(diffbits) = 2*(2-indneq(diffbits))-1; % indneq = 1/2 --> +1/-1
    y2p(diffbits) = 2*(indneq(diffbits)-1)-1; % indneq = 1/2 --> -1/+1
    samebits = find(y1p == 0);
    y1p(samebits) = 2*(2-indeq(samebits))-1;
    y2p(samebits) = 2*(2-indeq(samebits))-1;

    nonzero_grad_1 = sum(abs(y1-y1p)) ~= 0;
    nonzero_grad_2 = sum(abs(y2-y2p)) ~= 0;
    nonzero_grad = nonzero_grad_1 | nonzero_grad_2;

    tmp = [x1(:,nonzero_grad_1) * (y1(:,nonzero_grad_1) - y1p(:,nonzero_grad_1))' + ...
	   x2(:,nonzero_grad_2) * (y2(:,nonzero_grad_2) - y2p(:,nonzero_grad_2))']';

    if (verbose)
      n_bits_on    = n_bits_on    + sum(y1==1, 2) + sum(y2==1, 2);
      n_bits_total = n_bits_total + 2*ncases;
      
      r = n_bits_on / n_bits_total;
      bits_useless   = (min(r, 1-r)) < .03;
      n_bits_useless = sum(bits_useless);
      
	  hdis = nb-(nb/2+(sum(y1.*y2)/2));
      TP = TP + sum((hdis <= rho) & (l == 1));
      TN = TN + sum((hdis >  rho) & (l == 0));
      FP = FP + sum((hdis <= rho) & (l == 0));
      FN = FN + sum((hdis >  rho) & (l == 1));
    
      nnz = nnz + (sum(nonzero_grad_1)+sum(nonzero_grad_1))/2;
      nnz_fn = nnz_fn + sum(nonzero_grad & (hdis >  rho) & (l == 1));
      nnz_fp = nnz_fp + sum(nonzero_grad & (hdis <= rho) & (l == 0));
      nnz_tp = nnz_tp + sum(nonzero_grad & (hdis <= rho) & (l == 1));
      nnz_tn = nnz_tn + sum(nonzero_grad & (hdis >  rho) & (l == 0));
    
      batch_bound =  sum(sum(W .* -tmp)) + sum(loss(nflip(:)+1+repmat(nb+1,[ncases,1]).*(0:ncases-1)'));
      bound = bound + batch_bound;
    end

    Winc = momentum * Winc + eta * (tmp / ncases - shrink_w * [W(:,1:end-1) zeros(nb, 1)]);
    if (zerobias)
      Winc(:,end) = 0;
    end  
    W = W + Winc;
    % we don't re-normalize rows of W as mentioned in the paper anymore, instead we use weight decay
    % i.e., L2 norm regularizer
    % normW = [repmat(sqrt(sum(W(:,1:end-1).^2,2)), [1 dtr+1])]; 
    % W = W./normW;
  end
  
  fprintf('(%d/%.3f) ', t, eta);

  if (~verbose || (~param.doval_during && ~param.doval_after) || (verbose && mod(t, verbose) ~= 0))
    fprintf('\r');
  end

  if (verbose && t < maxiter && mod(t, verbose) == 0)
    fprintf([' bound:%.3f----nnz:%.1f  FN[%.0f%% %d]  FP[%.0f%% %d]  TP[%.0f%% %d]  TN[%.0f%% %d]' ...
    	     '----prec:%.3f  rec:%.3f  norm W=%.2f | nsm:%d (%.3f)'], bound/maxb, nnz/maxb, 100*nnz_fn/FN, FN, 100*nnz_fp/FP, ...
    	    FP, 100*nnz_tp/TP, TP, 100*nnz_tn/TN, TN, 100*TP/(TP+FP), TP/(TP+FN), sqrt(sum(W(:).^2)), ...
    	    n_bits_useless, min(min(r, 1 - r)));
    fprintf('\n');
    
    nnz_fp = 0; nnz_fn = 0; nnz_tp = 0; nnz_tn = 0;
    FP = 0; TP = 0; FN = 0; TN = 0;
    sum_resp = 0;
    n_sum_resp = 0;
  
    Wall{t+1} = W;
  end
  nnz = 0;
  
  if(param.doval_during && t < maxiter)
    if (mod(t, floor(maxiter/(param.doval_during))) == 0)
      eval(t, eta, data, W, rho, verbose);
    end
  end
  
  if(param.doval_after && t >= maxiter)
    [ap err] = eval(t, eta, data, W, rho, verbose);
    avg_err = avg_err + err;
    avg_ap  = avg_ap  + ap;
  end
end

if (param.doval_after)
  avg_ap  = avg_ap  / param.doval_after;
  avg_err = avg_err / param.doval_after;
  if (verbose)
    fprintf('Mean AP over last %d steps: %.3f                         \n', param.doval_after, avg_ap);
    if (strcmp(data.MODE, 'sem-full-mnist')) % mnist
      fprintf('Mean error over last %d steps: %.3f                         \n', param.doval_after, avg_err);
    end
  end
end

param.ap  = avg_ap;
param.err = avg_err;
best_W = W;
best_params = param;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ap err] = eval(t, eta, data, W, rho, verbose)

err = NaN;
if (strcmp(data.MODE, 'sem-full-mnist')) %mnist
  err = eval_mnist(W, data);
  fprintf('~~~ err: %.3f ', err);
end

[p1 r1] = eval_linear_hash(W, data);
p1(isnan(p1)) = 1;

ap = 0;
if (strcmp(data.MODE, 'sem-22k-labelme'))
  % semantic labelme
  [ap pcode] = eval_labelme(W, data);
  if (verbose)
    fprintf('~~~ prec(rho=%d): %.3f ~~~ recall(rho=%d): %.4f ~~~ ap: %.5f ~~~ ap(50): %.3f ~~~ ap(100): %.3f ~~~ ap(500): %.3f\n', ...
	    rho, p1(rho+1), rho, r1(rho+1), ap, pcode(50), pcode(100), pcode(500));
  end
else
  ap = sum([(p1(1:end-1)+p1(2:end))/2].*[(r1(2:end)-r1(1:end-1))]);
  if (verbose)
    fprintf('~~~ prec(rho=%d): %.3f ~~~ recall(rho=%d): %.4f ~~~ ap: %.5f\n', ...
	    rho, p1(rho+1), rho, r1(rho+1), ap);
  end
end
