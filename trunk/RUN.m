%%%%%%%%%%%%%%%%%%%% Small Datasets %%%%%%%%%%%%%%%%%%%%

% Whether re-create the datasets with new train/test subsets or load it from existing mat files
recreate_data = 0;

for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'}
  mode = modei{1}

  if (recreate_data)
    % re-create the datasets with new train/test subsets
    % data available at http://www.eecs.berkeley.edu/~kulis/data/
    data = create_data('kulis', mode);
    % creates the data structure with training and test points
    % save the data structure if you want
    data_pca = do_pca(data, 40);
    % performs PCA dimentionality reduction to retain a 40D subspace
  else  
    % loading data structures' mat files with pre-defined train/test sbusets used in MLH paper 
    % data available at http://www.cs.toronto.edu/~norouzi/get/small-datasets.tar
    load(['data/kulis/', mode]);
  end

%  nbs = [10 15 20 25 30 35 40 45 50];
  nbs = [10 20 30 40 50];
  for nb = nbs
    
    % assumes lambda = 1 / so no validation on lambda
    % assumes rho = 3 / so no validation on rho
    % learning rate is also fixed at .1
    % validation for the weight decay parameter
    fprintf('[nb = %d]\n', nb);
    t0 = tic;
    Wtmp{nb} = MLH(data_pca, 'hinge', nb, [.1], .9, [1 1], 100, 'train', 20, 3, 1, 1, 1, [.1 .03 .01 .003 .001 .0003 .0001], 1);
    time_validation(nb) = toc(t0)
  end
  
  clear pmlh rmlh W;
  ntrials = 10; % number of trials (the same train/test selection is used)
  for i=1:ntrials
    fprintf('[%d / %d]\n', i, ntrials);
    for nb = nbs
      t1 = tic;
      [m ind] = max([Wtmp{nb}.ap]); % best setting according to evaluation
      W{i, nb} = MLH(data_pca, 'hinge', nb, [.1], .9, [Wtmp{nb}(ind).params.ratio_loss_pos Wtmp{nb}(ind).params.ratio_loss_neg], 100, 'trainval', 50, 3, 1, 0, 0, Wtmp{nb}(ind).params.shrink_w, 1);
      time_test(i, nb) = toc(t1);
    end
  
    for nb = nbs
      pmlh(i, nb, :) = zeros(1, 51);
      rmlh(i, nb, :) = zeros(1, 51);
      [pmlh(i, nb, 1:nb+1) rmlh(i, nb, 1:nb+1)] = eval_linear_hash(W{i, nb}.W, data_pca);
    end
    save(['res/mlh_', mode, '.mat'], 'pmlh', 'rmlh', 'mode', 'W', 'Wtmp', 'time_test', 'time_validation', 'ntrials');
  end
end



%%%%%%%%%%%%%%%%%%%% Plots for Small Datasets %%%%%%%%%%%%%%%%%%%%

% Precision curves & Recall curves for certain radius of R as a function of code length
for R = [4];
for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'}

  mode = modei{1}
  load(['res/mlh_', mode, '.mat']);
  load(['../res/SH_', mode, '.mat']);
  load(['../res/LSH_', mode, '.mat']);
  load(['../res/BRE_', mode, '.mat']);
  % load(['../res/SIKH_', mode, '.mat']);
  
  pmlh_std = squeeze(std(pmlh,0,1));
  pmlh_mean = squeeze(mean(pmlh,1));
  psh_std = squeeze(std(psh,0,1));
  psh_mean = squeeze(mean(psh,1));
  plsh_std = squeeze(std(plsh,0,1));
  plsh_mean = squeeze(mean(plsh,1));
  pbre_std = squeeze(std(pbre,0,1));
  pbre_mean = squeeze(mean(pbre,1));
  % psikh_std = squeeze(std(psikh,0,1));
  % psikh_mean = squeeze(mean(psikh,1));

  rmlh_std = squeeze(std(rmlh,0,1));
  rmlh_mean = squeeze(mean(rmlh,1));
  rsh_std = squeeze(std(rsh,0,1));
  rsh_mean = squeeze(mean(rsh,1));
  rlsh_std = squeeze(std(rlsh,0,1));
  rlsh_mean = squeeze(mean(rlsh,1));
  rbre_std = squeeze(std(rbre,0,1));
  rbre_mean = squeeze(mean(rbre,1));
  % rsikh_std = squeeze(std(rsikh,0,1));
  % rsikh_mean = squeeze(mean(rsikh));

  load(['../res/my2_', mode, '.mat']);
  pmy_std = squeeze(std(pmy,0,1));
  pmy_mean = squeeze(mean(pmy,1));
  rmy_std = squeeze(std(rmy,0,1));
  rmy_mean = squeeze(mean(rmy,1));

  [size(pmlh,1) size(psh,1) size(plsh,1) size(pbre,1)]

%  nbs_for_plot = [10 15 20 25 30 35 40 45 50];
  nbs_for_plot = [10 20 30 40 50];
  
  cap.tit = [mode, ' (precision)'];
  cap.xlabel = ['Code length (bits)'];
  cap.ylabel = ['Precision for Hamm. dist. <= ', num2str(R-1)];
  p=[pmlh_mean(:,R) pbre_mean(:,R) plsh_mean(:,R) psh_mean(:,R)];  % psikh_mean(:,R)];
  e=[pmlh_std(:,R)  pbre_std(:,R)  plsh_std(:,R)  psh_std(:,R) ];  % psikh_std(:,R)];
  fig = make_err_plot(repmat(nbs_for_plot', [1 4]), p(nbs_for_plot, :), e(nbs_for_plot, :), ...
		      {'MLH', 'BRE', 'LSH', 'SH'}, cap, 'br', 1);
%  exportfig(fig, ['figs/new_',mode,'-prec-',num2str(R-1),'.eps'], 'Color', 'rgb');

  cap.tit = [mode, ' (recall)'];
  cap.xlabel = ['Number of bits'];
  cap.ylabel = ['Recall for Hamm. dist. <= ', num2str(R-1)];
  r =  [rmlh_mean(:,R) rbre_mean(:,R) rlsh_mean(:,R) rsh_mean(:,R)]; % rsikh_mean(:,R)];
  er = [rmlh_std(:,R)  rbre_std(:,R)  rlsh_std(:,R)  rsh_std(:,R) ]; % psikh_std(:,R)];
  fig = make_err_plot(repmat(nbs_for_plot', [1 4]), r(nbs_for_plot, :), er(nbs_for_plot, :), ...
  		     {'MLH', 'BRE', 'LSH', 'SH'}, cap, 'tr', 1);

  r =  [rmy_mean(:,R) rmlh_mean(:,R)];
  er = [rmy_std(:,R)  rmlh_std(:,R) ];
  fig = make_err_plot(repmat(nbs_for_plot', [1 2]), r(nbs_for_plot, :), er(nbs_for_plot, :), ...
  		     {'MLH', 'MY'}, cap, 'tr', 1);
  
%  exportfig(fig, ['figs/new_',mode,'-recall-',num2str(R-1),'.eps'], 'Color', 'rgb');
end
end

% Precision-Recall curves as a function of code length
for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'} 
  mode = modei{1}
  load(['../res/my2_', mode, '.mat']);
  load(['../res/SH_', mode, '.mat']);
  load(['../res/LSH_', mode, '.mat']);
  load(['../res/BRE_', mode, '.mat'])
  % load(['../res/SIKH_', mode, '.mat']);
  load(['res/mlh_', mode, '.mat']);

  for nb = [30, 50];
    clear precs_my precs_lsh precs_bre precs_lsh precs_sh precs_sikh precs_mlh;
    recs_my = [max(rmy(:,nb,1)):.02:min(rmy(:,nb,nb+1)), min(rmy(:,nb,nb+1))];
    for i=1:10
      precs_my(i,:) = compute_prec_at_recall(squeeze(rmy(i,nb,1:nb+1)), squeeze(pmy(i,nb,1:nb+1)), recs_my);
    end

    recs_mlh = [max(rmlh(:,nb,1)):.02:min(rmlh(:,nb,nb+1)), min(rmlh(:,nb,nb+1))];
    for i=1:size(pmlh, 1)
      precs_mlh(i,:) = compute_prec_at_recall(squeeze(rmlh(i,nb,1:nb+1)), squeeze(pmlh(i,nb,1:nb+1)), recs_mlh);
    end

    recs_bre = [max(rbre(:,nb,1)):.02:min(rbre(:,nb,nb+1)), min(rbre(:,nb,nb+1))];
    for i=1:10
      precs_bre(i,:) = compute_prec_at_recall(squeeze(rbre(i,nb,1:nb+1)), squeeze(pbre(i,nb,1:nb+1)), recs_bre);
    end

    recs_lsh = [max(rlsh(:,nb,1)):.02:min(rlsh(:,nb,nb+1)), min(rlsh(:,nb,nb+1))];
    for i=1:10
      precs_lsh(i,:) = compute_prec_at_recall(squeeze(rlsh(i,nb,1:nb+1)), squeeze(plsh(i,nb,1:nb+1)), recs_lsh);
    end

    recs_sh = [max(rsh(:,nb,1)):.02:min(rsh(:,nb,nb+1)), min(rsh(:,nb,nb+1))];
    for i=1:10
      precs_sh(i,:) = compute_prec_at_recall(squeeze(rsh(i,nb,1:nb+1)), squeeze(psh(i,nb,1:nb+1)), recs_sh);
    end

    % recs_sikh = [max(rsikh(:,nb,1)):.02:min(rsikh(:,nb,nb+1)), min(rsikh(:,nb,nb+1))];
    % for i=1:10
    %   precs_sikh(i,:) = compute_prec_at_recall(squeeze(rsikh(i,nb,1:nb+1)), squeeze(psikh(i,nb,1:nb+1)), recs_sikh);
    % end

    cap.tit = [mode, ' (precision-recall) using ', num2str(nb), ' bits'];
    cap.xlabel = ['Recall'];
    cap.ylabel = ['Precision'];
    fig = make_err_plot({recs_mlh,            recs_bre,           recs_lsh,           recs_sh,           }, ...
		        {mean(precs_mlh,1),   mean(precs_bre,1),  mean(precs_lsh,1),  mean(precs_sh),    }, ...
		        {std(precs_mlh,0,1),  std(precs_bre,0,1), std(precs_lsh,0,1), std(precs_sh,0,1), }, ...
		        {'MLH',               'BRE',              'LSH',              'SH',              }, ...
			cap, 'tr', 1);

    cap.tit = [mode, ' (precision-recall) using ', num2str(nb), ' bits'];
    cap.xlabel = ['Recall'];
    cap.ylabel = ['Precision'];
    fig = make_err_plot({recs_my,            recs_mlh} , ...
		        {mean(precs_my,1),   mean(precs_mlh,1)}, ...
		        {std(precs_my,0,1),  std(precs_mlh,0,1)}, ...
		        {'MY', 'MLH'}, cap, 'tr', 1);
    % exportfig(fig, ['figs/', mode, '-prec-recall-', num2str(nb), '.eps'], 'Color', 'rgb');
  end
end



%%%%%%%%%%%%%%%%%%%% Euclidean 22K LabelMe %%%%%%%%%%%%%%%%%%%%

data = create_data('euc-22K-labelme', 100);
data_pca =  do_pca(data, 40);

clear Wtmp_size_batch Wtmp Wtmp2 Wtmp_rho;
clear pmlh rmlh;

nbs = [32 64 128 256];
for i = 1:2; % :10
for nb = nbs
  nb
  
  if (~exist('best_params') || numel(best_params) < nb || best_params(nb).rho == 0)
    % Do validation
    if (~exist('best_params') || ~isfield(best_params, 'rho') || isempty(best_params(nb/2).rho))
      last_rho = ceil(2*(nb/2)/16);
    else
      last_rho = best_params(nb/2).rho;
    end

    best_params(nb).size_batches = 100;
    best_params(nb).eta = .03;
    best_params(nb).rho = last_rho * 2;

    % validation for lambda in hinge loss
    Wtmp2{nb} = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [2.5 .5; 2 .5; 1.5 .5; 1.4 .7; 1 1], 100, 'train', 50, best_params(nb).rho, ...
		    1, 1, 1, .01, 1);
    best_ap = -1;
    for j = 1:numel(Wtmp2{nb})
      fprintf('%.3f %.2f %.2f %.2f\n', Wtmp2{nb}(j).ap, Wtmp2{nb}(j).params.ratio_loss_pos, Wtmp2{nb}(j).params.ratio_loss_neg, Wtmp2{nb}(j).params.eta);
      if (Wtmp2{nb}(j).ap > best_ap)
	best_ap = Wtmp2{nb}(j).ap;
        best_params(nb).ratio_loss_pos = Wtmp2{nb}(j).params.ratio_loss_pos;
        best_params(nb).ratio_loss_neg = Wtmp2{nb}(j).params.ratio_loss_neg;
      end
    end
  
    % validation for weight decay parameter
    Wtmp_shrink_w = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [best_params(nb).ratio_loss_pos best_params(nb).ratio_loss_neg], 100, 'train', ...
			50, best_params(nb).rho, 1, 1, 1, [.1 .03 .01 .003 .001], 1);
    best_ap = -1;
    for j = 1:numel(Wtmp_shrink_w)
      if (Wtmp_shrink_w(j).ap > best_ap)
        best_ap = Wtmp_shrink_w(j).ap;
        best_params(nb).shrink_w = Wtmp_shrink_w(j).params.shrink_w;
      end
      fprintf('%.3f %d\n', Wtmp_shrink_w(j).ap, Wtmp_shrink_w(j).params.shrink_w);
    end
    fprintf('The best weight decay(%d) = %d\n', nb, best_params(nb).shrink_w);
  
    % validation for rho in hinge loss
    % rho might get large. If a certain retrieval hamming radius at test-time is desired, rho
    % should be set manually.
    rhos = [last_rho*2-2 last_rho*2-1 last_rho*2 last_rho*2+1 last_rho*2+2]; % should be sorted!
                                                                             % (see below)
    rhos(rhos < 1) = [];
    Wtmp_rho = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [best_params(nb).ratio_loss_pos best_params(nb).ratio_loss_neg], 100, 'train', 100, rhos, 1, 1, 1, .01, 1); 
    best_ap = -1;
    for j = 1:numel(Wtmp_rho)
      if (Wtmp_rho(j).ap > best_ap + .01) % only if average precision gets better by 1 percent, it
					  % is worth it to increase rho
	best_ap = Wtmp_rho(j).ap;
	best_params(nb).rho = Wtmp_rho(j).params.rho;
      end
      fprintf('%.3f %d\n', Wtmp_rho(j).ap, Wtmp_rho(j).params.rho);
    end
    fprintf('The best rho(%d) = %d\n', nb, best_params(nb).rho);
  end
  
  % validation for eta (learning rate)
  % Wtmp_eta = MLH(data_pca, 'hinge', nb, [.1 .03 .01], .9, [1 1], 100, 'train', 50, last_rho*2, 1, 1, 1, .01, 1);
  % best_ap = -1;
  % for j = 1:numel(Wtmp_eta)
  %   if (Wtmp_eta(j).ap > best_ap)
  %     best_ap = Wtmp_eta(j).ap;
  %     best_params(nb).eta = Wtmp_eta(j).params.eta;
  %   end
  %   fprintf('%.3f %d\n', Wtmp_eta(j).ap, Wtmp_eta(j).params.eta);
  % end
  % fprintf('The best batch size(%d) = %d\n', nb, best_params(nb).eta);
  
  best_params(nb)
  W{i, nb} = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [best_params(nb).ratio_loss_pos best_params(nb).ratio_loss_neg], 100, 'trainval', ...
  		 2000, best_params(nb).rho, 1, 1, 0, best_params(nb).shrink_w, 1);
  pmlh(i, nb, :) = zeros(1, max(nbs)+1);
  rmlh(i, nb, :) = zeros(1, max(nbs)+1);
  [pmlh(i, nb, 1:nb+1) rmlh(i, nb, 1:nb+1)] = eval_linear_hash(W{i, nb}.W, data_pca);
  save res/mlh_euc-22K-labelme pmlh rmlh best_params W;
end
end



%%%%%%%%%% PLOTS %%%%%%%%%%%

load ../res2/my_labelme-euc2;
load res/mlh_euc-22K-labelme.mat

nbs = [256];
for nb = nbs;
  clear precs_mlh precs_my precs_lsh precs_bre2 precs_lsh precs_sh precs_sikh precs_bre_my precs_my_half;

  n_mlh = 1;
  recs_mlh = [max(rmlh(1:n_mlh,nb,1)):.02:min(rmlh(1:n_mlh,nb,nb+1)), min(rmlh(1:n_mlh,nb,nb+1))];
  for i=1:n_mlh %:10 TODO: rerun the experiments
    precs_mlh(i,:) = compute_prec_at_recall(squeeze(rmlh(i,nb,1:nb+1)), squeeze(pmlh(i,nb,1:nb+1)), recs_mlh);
  end
  
  n_my = 5;  
  recs_my = [max(rmy(1:n_my,nb,1)):.02:min(rmy(1:n_my,nb,nb+1)), min(rmy(1:n_my,nb,nb+1))];
  for i=1:n_my %:10 TODO: rerun the experiments
    precs_my(i,:) = compute_prec_at_recall(squeeze(rmy(i,nb,1:nb+1)), squeeze(pmy(i,nb,1:nb+1)), recs_my);
  end

  cap.tit = [''];
  cap.xlabel = ['Recall'];
  cap.ylabel = ['Precision'];
  fig = make_err_plot(...
      {recs_my, recs_mlh}, ...
      {mean(precs_my,1), mean(precs_mlh,1)}, ...
      {std(precs_my,0,1), std(precs_mlh,0,1)}, ...
      {'MY', 'MLH'}, cap, 'tr', 1);  
  
  nbre_my = numel([Wbre{:,nb}]);
  [n_my nbre_my sum(pbre2(:,nb,1) > 0)]

  recs_my_half = [max(rmy(1:nmy,nb/2,1)):.02:min(rmy(1:nmy,nb/2,nb/2+1)), min(rmy(1:nmy,nb/2,nb/2+1))];
  for i=1:numel([W{:,nb}]) %:10 TODO: rerun the experiments
    precs_my_half(i,:) = compute_prec_at_recall(squeeze(rmy(i,nb/2,1:nb/2+1)), squeeze(pmy(i,nb/2,1:nb/2+1)), recs_my_half);
  end
  
  recs_lsh = [max(rlsh(:,nb,1)):.02:min(rlsh(:,nb,nb+1)), min(rlsh(:,nb,nb+1))];
  for i=1:10
    precs_lsh(i,:) = compute_prec_at_recall(squeeze(rlsh(i,nb,1:nb+1)), squeeze(plsh(i,nb,1:nb+1)), recs_lsh);
  end
  recs_sh = [max(rsh(:,nb,1)):.02:min(rsh(:,nb,nb+1)), min(rsh(:,nb,nb+1))];
  for i=1 % SH is deterministic
    precs_sh(i,:) = compute_prec_at_recall(squeeze(rsh(i,nb,1:nb+1)), squeeze(psh(i,nb,1:nb+1)), recs_sh);
  end
  recs_sikh = [max(rsikh(:,nb,1)):.02:min(rsikh(:,nb,nb+1)), min(rsikh(:,nb,nb+1))];
  for i=1:10
    precs_sikh(i,:) = compute_prec_at_recall(squeeze(rsikh(i,nb,1:nb+1)), squeeze(psikh(i,nb,1:nb+1)), recs_sikh);
  end
  nbre2 = sum(pbre2(:,nb,1) > 0);
  recs_bre2 = [max(rbre2(1:nbre2,nb,1)):.02:min(rbre2(1:nbre2,nb,nb+1)), min(rbre2(1:nbre2,nb,nb+1))];
  for i=1:sum(pbre2(:,nb,1) > 0)% TODO: rerun the experiments
    precs_bre2(i,:) = compute_prec_at_recall(squeeze(rbre2(i,nb,1:nb+1)), squeeze(pbre2(i,nb,1:nb+1)), recs_bre2);
  end
  recs_bre_my = [max(rbre(1:nbre_my,nb,1)):.02:min(rbre(1:nbre_my,nb,nb+1)), min(rbre(1:nbre_my,nb,nb+1))];
  for i=1:numel([Wbre{:,nb}]); %:10 TODO: rerun the experiments
    precs_bre_my(i,:) = compute_prec_at_recall(squeeze(rbre(i,nb,1:nb+1)), squeeze(pbre(i,nb,1:nb+1)), recs_bre_my);
  end

precs_sh = repmat(precs_sh, [2 1]);
%precs_my = repmat(precs_my, [2 1]); % REMOVE Late
precs_bre2 = repmat(precs_bre2, [2 1]); % REMOVE Late
%precs_bre_my = repmat(precs_bre_my, [2 1]); % REMOVE Late

cap.tit = [''];
cap.xlabel = ['Recall'];
cap.ylabel = ['Precision'];
if (nb > 16)
  fig = makePrecPlot({recs_my, recs_bre2, recs_lsh, recs_sh, recs_sikh, recs_bre_my, recs_my_half}, {mean(precs_my), mean(precs_bre2), mean(precs_lsh), mean(precs_sh), mean(precs_sikh), mean(precs_bre_my), mean(precs_my_half)}, {std(precs_my), std(precs_bre2), std(precs_lsh), std(precs_sh), std(precs_sikh), std(precs_bre_my), std(precs_my_half)}, {'MLH', 'BRE', 'LSH', 'SH', 'SIKH', 'BRE2', 'MLH.5'}, cap, 'tr', 1);
else
  fig = makePrecPlot(...
      {recs_my, recs_bre2, recs_lsh, recs_sh, recs_sikh, recs_bre_my}, ...
      {mean(precs_my), mean(precs_bre2), mean(precs_lsh), mean(precs_sh), mean(precs_sikh), mean(precs_bre_my)}, ...
      {std(precs_my), std(precs_bre2), std(precs_lsh), std(precs_sh), std(precs_sikh), std(precs_bre_my)}, ...
      {'MLH', 'BRE', 'LSH', 'SH', 'SIKH', 'BRE2'}, cap, 'tr', 1);
mode='large-labelme-euc';
end
exportfig(fig, ['icml11/figs/new_',mode,'-prec-recall', num2str(nb),'.eps'], 'Color', 'rgb');
%saveas(fig, ['icml11/figs/',mode,'-prec2-recall', num2str(nb),'.eps'], 'psc2');
end
