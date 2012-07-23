function [p1 r1] = eval_linear_hash(W, data)

Ntest = data.Ntest;
Xtest = data.Xtest;
StestTraining = data.StestTraining;

Ntraining = data.Ntraining;
Xtraining = data.Xtraining;

B1 = W * [Xtraining; ones(1,Ntraining)] > 0;
B2 = W * [Xtest; ones(1,Ntest)] > 0;
B1 = compactbit(B1);
B2 = compactbit(B2);

Dhamm = hammDist_mex(B2, B1);		% a mex version of hammDist
% Dhamm = hammDist(B2, B1);

[p1 r1] = evaluation3(StestTraining, Dhamm, size(W,1));
p1 = p1';

p1 = full(p1);
r1 = full(r1);
