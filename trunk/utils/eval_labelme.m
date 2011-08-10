function [ap p_code] = evalLabelme(W, data);

B1 = (W*[data.Xtraining ones(data.Ntraining, 1)]' > 0)';
B2 = (W*[data.Xtest ones(data.Ntest, 1)]' > 0)';

ndxtrain = 1:data.Ntraining;
ndxtest = data.Ntraining+1:data.Ntraining+data.Ntest;
code(:,ndxtrain) = B1'; %data.label_train_code;
code(:,ndxtest)  = B2'; %data.label_test_code;
code = compactbit(code')';
code = uint8(code);

P_code = zeros(numel(ndxtrain), numel(ndxtest));
for n = 1:length(ndxtest)
    ndx = ndxtest(n);

    % compute your distance
    D_code = hammD2(code(:,ndx),code(:,ndxtrain));
    [foo, j_code] = sort(D_code, 'ascend'); % I assume that smaller distance means closer
    j_code = ndxtrain(j_code);

    % get groundtruth sorting
    D_truth = data.DtestTraining(ndx-data.Ntraining,:);
    [foo, j_truth] = sort(D_truth);
    j_truth = ndxtrain(j_truth);

    % evaluation
    [h, P_code(:,n)] = neighborsRecall(j_truth, j_code, 'r');
end

p_code = mean(P_code,2);
ap = mean(p_code(1:data.max_care));
