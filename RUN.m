data = create_data('euc-22K-labelme', 100);
data_pca =  do_pca(data, 40);

clear Wtmp_size_batch Wtmp Wtmp2 Wtmp_rho;
clear pmlh rmlh;

nbs = [16 32 64 128 256];
for i = 1:2; %2:10
for nb = nbs
  nb
  
  if (~exist('best_params') || numel(best_params) < nb || best_params(nb).rho == 0)
    % Do cross-validation
    
    if (~exist('best_params') || ~isfield(best_params, 'rho') || isempty(best_params(nb/2).rho))
      last_rho = ceil(2*(nb/2)/16);
    else
      last_rho = best_params(nb/2).rho;
    end

    best_params(nb).size_batches = 100;
    best_params(nb).eta = .03;

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

    rhos = [last_rho*2-2 last_rho*2-1 last_rho*2 last_rho*2+1 last_rho*2+2 last_rho*2+3];
    rhos(rhos < 1) = [];
    Wtmp_rho = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [1 1], 100, 'train', 50, rhos, 1, 1, 1, .01, 1); 
    best_ap = -1;
    for j = 1:numel(Wtmp_rho)
      if (Wtmp_rho(j).ap > best_ap)
	best_ap = Wtmp_rho(j).ap;
	best_params(nb).rho = Wtmp_rho(j).params.rho;
      end
      fprintf('%.3f %d\n', Wtmp_rho(j).ap, Wtmp_rho(j).params.rho);
    end
    fprintf('The best rho(%d) = %d\n', nb, best_params(nb).rho);

    Wtmp2{nb} = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [2 .5; 1.5 .5; 1.4 .7; 1 1; .7 1.4; .5 1.5; .5 2], 100, 'train', 50, best_params(nb).rho, ...
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
    
    Wtmp_shrink_w = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [best_params(nb).ratio_loss_pos best_params(nb).ratio_loss_neg], 100, 'train', ...
			50, best_params(nb).rho, 1, 1, 1, [.1 .03 .01 .003 .001 .0003 .0001], 1);
    best_ap = -1;
    for j = 1:numel(Wtmp_shrink_w)
      if (Wtmp_shrink_w(j).ap > best_ap)
        best_ap = Wtmp_shrink_w(j).ap;
        best_params(nb).shrink_w = Wtmp_shrink_w(j).params.shrink_w;
      end
      fprintf('%.3f %d\n', Wtmp_shrink_w(j).ap, Wtmp_shrink_w(j).params.shrink_w);
    end
    fprintf('The best weight decay(%d) = %d\n', nb, best_params(nb).shrink_w);
  end
  
  best_params(nb)
  W{i, nb} = MLH(data_pca, 'hinge', nb, [best_params(nb).eta], .9, [best_params(nb).ratio_loss_pos best_params(nb).ratio_loss_neg], 100, 'trainval', ...
		 2000, best_params(nb).rho, 1, 1, 1, best_params(nb).shrink_w, 1);
  pmlh(i, nb, :) = zeros(1, max(nbs)+1);
  rmlh(i, nb, :) = zeros(1, max(nbs)+1);
  [pmlh(i, nb, 1:nb+1) rmlh(i, nb, 1:nb+1)] = eval_linear_hash(W{i, nb}.W, data_pca);
  save res/mlh_euc-22K-labelme pmlh rmlh best_params W;
end
end

for modei = {'peekaboom', 'nursery', 'notredame', '10d', 'labelme', 'mnist'}
  mode = modei{1}
  data = create_data('kulis', mode);
  % creates the data structure with training and test points
  data_pca = do_pca(data, 40);
  % performs PCA dimentionality reduction to retain 40D subspace


  nbs = [5 10 15 20 25 30 35 40 45 50];
  for nb = nbs
    fprintf('[nb = %d]\n', nb);
    t0 = tic;
    Wtmp{nb} = MLH(data_pca, 'hinge', nb, [.1], .9, [1 1], 100, 'train', 20, 3, 1, 1, 1, [.1 .03 .01 .003 .001 .0003 .0001], 1);
    time_validation(nb) = toc(t0)
  end

  clear pmlh rmlh W;
  % how many trials
  ntrials = 10;
  for i=1:ntrials
    fprintf('[%d / %d]\n', i, ntrials);
    for nb = nbs
      t1 = tic;
      [m ind] = max([Wtmp{nb}.ap]); % best setting according to evaluation
      W{i, nb} = MLH(data_pca, 'hinge', nb, [.1], .9, [Wtmp{nb}(ind).params.ratio_loss_pos Wtmp{nb}(ind).params.ratio_loss_neg], 100, 'trainval', 20, 3, 1, 0, 1, Wtmp{nb}(ind).params.shrink_w, 1);
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

%%%%%%%%%% PLOTS %%%%%%%%%%%

for xth = [4];
for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'}

  mode = modei{1}
  load(['../res/my2_', mode, '.mat']);
  load(['../res/SH_', mode, '.mat']);
  load(['../res/LSH_', mode, '.mat']);
  load(['../res/BRE_', mode, '.mat']);
  load(['../res/SIKH_', mode, '.mat']);
  load(['res/mlh_', mode, '.mat']);
  
  pmy_std = squeeze(std(pmy));
  pmy_mean = squeeze(mean(pmy,1));
  psh_std = squeeze(std(psh));
  psh_mean = squeeze(mean(psh,1));
  plsh_std = squeeze(std(plsh));
  plsh_mean = squeeze(mean(plsh,1));
  pbre_std = squeeze(std(pbre));
  pbre_mean = squeeze(mean(pbre,1));
  % psikh_std = squeeze(std(psikh));
  % psikh_mean = squeeze(mean(psikh));
  % pblsh_mean = squeeze(mean(pblsh,1));
  % pblsh_std = squeeze(std(pblsh));
  pmlh_std = squeeze(std(pmlh));
  pmlh_mean = squeeze(mean(pmlh,1));

  rmy_std = squeeze(std(rmy));
  rmy_mean = squeeze(mean(rmy,1));
  rsh_std = squeeze(std(rsh));
  rsh_mean = squeeze(mean(rsh,1));
  rlsh_std = squeeze(std(rlsh));
  rlsh_mean = squeeze(mean(rlsh,1));
  rbre_std = squeeze(std(rbre));
  rbre_mean = squeeze(mean(rbre,1));
  % rsikh_std = squeeze(std(rsikh));
  % rsikh_mean = squeeze(mean(rsikh));
  % rblsh_mean = squeeze(mean(rblsh,1));
  % rblsh_std = squeeze(std(rblsh));
  rmlh_std = squeeze(std(rmlh));
  rmlh_mean = squeeze(mean(rmlh,1));

  [size(pmy,1) size(psh, 1) size(plsh, 1) size(pbre, 1) size(psikh, 1) size(pmlh,1)]

  nbs_for_plot = [10 15 20 25 30 35 40 45 50];
  
  cap.tit = [mode, ' (precision)'];
  cap.xlabel = ['Code length (bits)'];
  cap.ylabel = ['Precision for Hamm. dist. <= ', num2str(xth-1)];
  p=[pmy_mean(:,xth) pbre_mean(:,xth) plsh_mean(:,xth) psh_mean(:,xth) pmlh_mean(:,xth)]; % psikh_mean(:,xth)];
  e=[pmy_std(:,xth)  pbre_std(:,xth)  plsh_std(:,xth)  psh_std(:,xth)  pmlh_std(:,xth)];  % psikh_std(:,xth)];
  fig = make_err_plot(repmat(nbs_for_plot', [1 5]), p(nbs_for_plot, :), e(nbs_for_plot, :), {'MY', 'BRE', 'LSH', 'SH', 'MLH'}, cap, 'br', 1);
  exportfig(fig, ['figs/new_',mode,'-prec-',num2str(xth-1),'.eps'], 'Color', 'rgb');

  cap.tit = [mode, ' (recall)'];
  cap.xlabel = ['Number of bits'];
  cap.ylabel = ['Recall of neighbors with Hamm. distance <= ', num2str(xth-1)];
  r =  [rmy_mean(:,xth) rbre_mean(:,xth) rlsh_mean(:,xth) rsh_mean(:,xth) rmlh_mean(:,xth)];% rsikh_mean(:,xth)];
  er = [rmy_std(:,xth)  rbre_std(:,xth)  rlsh_std(:,xth)  rsh_std(:,xth)  rmlh_std(:,xth)]; % psikh_std(:,xth)];
  fig = make_err_plot(repmat(nbs_for_plot', [1 5]), r(nbs_for_plot, :), er(nbs_for_plot, :), ...
  		     {'MY', 'BRE', 'LSH', 'SH', 'MLH'}, cap, 'tr', 1);
  exportfig(fig, ['figs/new_',mode,'-recall-',num2str(xth-1),'.eps'], 'Color', 'rgb');
end
end

for modei = {'labelme', 'mnist', 'peekaboom', 'nursery', 'notredame', '10d'} 
  mode = modei{1}
  load(['../res/my2_', mode, '.mat']);
  load(['../res/SH_', mode, '.mat']);
  load(['../res/LSH_', mode, '.mat']);
  load(['../res/BRE_', mode, '.mat'])
  %load(['../res/SIKH_', mode, '.mat']);
  load(['res/mlh_', mode, '.mat']);

  for nb = [30, 45];
    clear precs_my precs_lsh precs_bre precs_lsh precs_sh precs_sikh precs_mlh;
    recs_my = [max(rmy(:,nb,1)):.02:min(rmy(:,nb,nb+1)), min(rmy(:,nb,nb+1))];
    for i=1:10
      precs_my(i,:) = compute_prec_at_recall(squeeze(rmy(i,nb,1:nb+1)), squeeze(pmy(i,nb,1:nb+1)), recs_my);
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
    recs_sikh = [max(rsikh(:,nb,1)):.02:min(rsikh(:,nb,nb+1)), min(rsikh(:,nb,nb+1))];
    for i=1:10
      precs_sikh(i,:) = compute_prec_at_recall(squeeze(rsikh(i,nb,1:nb+1)), squeeze(psikh(i,nb,1:nb+1)), recs_sikh);
    end
    recs_mlh = [max(rmlh(:,nb,1)):.02:min(rmlh(:,nb,nb+1)), min(rmlh(:,nb,nb+1))];
    for i=1:2
      precs_mlh(i,:) = compute_prec_at_recall(squeeze(rmlh(i,nb,1:nb+1)), squeeze(pmlh(i,nb,1:nb+1)), recs_mlh);
    end

    cap.tit = [mode, ' (precision-recall) using ', num2str(nb), ' bits'];
    cap.xlabel = ['Recall'];
    cap.ylabel = ['Precision'];

    % fig = make_err_plot({recs_my,        recs_bre,        recs_lsh,        recs_sh,        recs_sikh}, ...
    % 		   {mean(precs_my), mean(precs_bre), mean(precs_lsh), mean(precs_sh), mean(precs_sikh)}, ...
    % 		   {std(precs_my),  std(precs_bre),  std(precs_lsh),  std(precs_sh),  std(precs_sikh)}, ...
    % 		   {'MLH', 'BRE', 'LSH', 'SH', 'SIKH'}, cap, 'tr', 1);

    fig = make_err_plot({recs_my,        recs_bre,        recs_lsh,        recs_sh,        recs_mlh} , ...
		       {mean(precs_my), mean(precs_bre), mean(precs_lsh), mean(precs_sh), mean(precs_mlh)}, ...
		       {std(precs_my),  std(precs_bre),  std(precs_lsh),  std(precs_sh),  std(precs_mlh)}, ...
		       {'MLH', 'BRE', 'LSH', 'SH', 'MLH'}, cap, 'tr', 1);

    exportfig(fig, ['figs/', mode, '-prec-recall-', num2str(nb), '.eps'], 'Color', 'rgb');
  end
end
