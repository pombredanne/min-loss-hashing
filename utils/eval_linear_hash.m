function [p1 r1] = eval_linear_hash(W, data, val)

Ntest = data.Ntest;
Xtest = data.Xtest;
WtrueTestTraining = data.WtrueTestTraining;

if (isfield(data, 'trainset'))
  if (strcmp(data.trainset, 'trainval'))
    fprintf('real test ');
  end
end

Ntraining = data.Ntraining;
Xtraining = data.Xtraining;

B1 = W * [Xtraining; ones(1,Ntraining)] > 0;
B2 = W * [Xtest; ones(1,Ntest)] > 0;
B1 = compactbit(B1);
B2 = compactbit(B2);
% Dhamm = hammingDist(B2, B1);
Dhamm = hammingDist2(B2, B1);		% a faster version of hammingDist
[p1 r1] = evaluation2(WtrueTestTraining, Dhamm, size(W,1)); %, 1, 'o-', 'color', 'r');
p1 = p1';

p1 = full(p1);
r1 = full(r1);
