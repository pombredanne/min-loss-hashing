function data = create_data(MODE, operand1, operand2, operand3)

if (strcmp(MODE, 'kulis'))
  fprintf('creating data %s - %s ...', MODE, operand1);
else
  fprintf('creating data %s ...', MODE);
end

data.MODE = MODE;

if (strcmp(MODE, 'uniform'))
  % Create a uniformly distributed synthesize dataset.
  % parameters
  dtr = operand1;
  Ntraining = 1000; % number training samples
  Ntest = 3000;     % number test samples
  averageNumberNeighbors = 50; % number of ground-truth neighbors for each training point (on average)
  aspectratio = ones(1, dtr);  % aspect ratio of different edge lenghts of the uniform hypercube

  Xtraining = rand([dtr, Ntraining]);
  for i=1:dtr
    Xtraining(i,:) = aspectratio(i) * Xtraining(i,:) - aspectratio(i)/2;
  end
  Xtest = rand([dtr, Ntest]);
  for i=1:dtr
    Xtest(i,:) = aspectratio(i)*Xtest(i,:) - aspectratio(i)/2;
  end

  data.aspectratio = aspectratio;
  data = construct_data(Xtraining, Xtest, [Ntraining, Ntest], averageNumberNeighbors, [], data); 
  % see bottom for construct_data(...)
  
elseif (strcmp(MODE, 'euc-22K-labelme'))
  % Create the Euclidean 22K labelme dataset.
  load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'gist');

  % data-points are stored in columns
  X = gist';
  clear gist;
  
  % center, then normalize data
  gist_mean = mean(X, 2);
  X = bsxfun(@minus, X, gist_mean);
  normX = sqrt(sum(X.^2, 1));
  X = bsxfun(@rdivide, X, normX);
  
  Xtraining = X(:, ndxtrain);
  Xtest = X(:, ndxtest);
  
  data = construct_data(Xtraining, Xtest, [numel(ndxtrain), numel(ndxtest)], operand1, operand2, data);
  % see bottom for construct_data(...)
  data.gist_mean = gist_mean;
  
elseif (strcmp(MODE, 'sem-22K-labelme'))
  % Create the semantic 22K labelme dataset.
  load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'DistLM', 'gist');
  
  % data-points are stored in columns
  X = gist';
  clear gist;

  % center, then normalize data
  gist_mean = mean(X, 2);
  X = bsxfun(@minus, X, gist_mean);
  normX = sqrt(sum(X.^2, 1));
  X = bsxfun(@rdivide, X, normX);
  if (~exist('operand3'))
    scale = 1;
  else
    scale = operand3;
  end
  X = X * scale;
  data.scale = scale;

  Xtraining = X(:, ndxtrain);
  Xtest     = X(:, ndxtest);
  Ntraining = numel(ndxtrain);
  Ntest = numel(ndxtest);

  DtrueTraining = -DistLM(ndxtrain, ndxtrain);
  DtrueTestTraining = -DistLM(ndxtest, ndxtrain);
  DistLM = (DistLM + DistLM') / 2; % making the distance symmetric
  DtrueTraining2 = -DistLM(ndxtrain, ndxtrain);
  DtrueTestTraining2 = -DistLM(ndxtest, ndxtrain);

  nNeighbors = operand1; % number of ground-truth neighbors for each training point (on average)
  Dball = sort(DtrueTraining2, 2); 
  Dballmin = mean(Dball(:, nNeighbors));

  data.MODE = MODE;
  data.Xtraining = Xtraining;
  data.Xtest = Xtest;
  data.WtrueTestTraining = DtrueTestTraining2 < Dballmin;
  data.Ntraining = Ntraining;
  data.Ntest = Ntest;
  data.Dballmin = Dballmin; 
  data.Dtraining = DtrueTraining;
  data.DtestTraining = DtrueTestTraining;
  data.averageNumberNeighbors = nNeighbors;
  data.max_care = operand2; % used for cross-validation in evalLabelme 

elseif (strcmp(MODE, 'kulis'))
  % From Brian Kulis's code; Preparing datasets from the BRE paper
  data.MODE = [MODE, ' - ', operand1];
  X = load(['data/kulis/', operand1, '.mtx'])';

  averageNumberNeighbors = 50; % number of ground-truth neighbors for each training point (on average)
  Ntraining = 1000;
  Ntest = min(3000, size(X,2)-1000);

  % center, then normalize data
  X = bsxfun(@minus, X, mean(X,2));
  normX = sqrt(sum(X.^2, 1));
  X = bsxfun(@rdivide, X, normX);
  
  % each time a new permuatation of data is used
  rp = randperm(size(X,2));
  trIdx = rp(1:Ntraining);
  testIdx = rp(Ntraining+1:Ntraining+Ntest);
  Xtraining = X(:, trIdx);
  Xtest = X(:, testIdx);
  
  data = construct_data(Xtraining, Xtest, [Ntraining, Ntest], averageNumberNeighbors, [], data);
  % see bottom for construct_data(...)
  
else
  error('The given mode is not recognized.\n');
end

fprintf('done\n');


function data = construct_data(Xtraining, Xtest, sizeSets, avgNNeighbors, proportionNeighbors, data)

% either avgNNeighbors or proportionNeighbors should be set. The other value should be empty ie., []
% avgNNeighbors is a number which determines the average number of neighbors for each data point
% proportionNeighbors is between 0 and 1 which determines the fraction of [similar pairs / total pairs]

[Ntraining, Ntest] = deal(sizeSets(1), sizeSets(2));
DtrueTraining = distMat(Xtraining);

if (~isempty(avgNNeighbors))
  sortedD = sort(DtrueTraining, 2);
  threshDist = mean(sortedD(:,avgNNeighbors));
  data.avgNNeighbors = avgNNeighbors;
else
  sortedD = sort(DtrueTraining(:));
  threshDist = sortedD(ceil(proportionNeighbors * numel(sortedD)));
  data.proportionNeighbors = proportionNeighbors;
end

DtrueTestTraining = distMat(Xtest, Xtraining); % size = [Ntest x Ntraining]

data.Xtraining = Xtraining;
data.Xtest = Xtest;  
data.WtrueTraining = DtrueTraining < threshDist;
data.WtrueTestTraining = DtrueTestTraining < threshDist;

data.Ntraining = Ntraining;
data.Ntest = Ntest;
data.threshDist = threshDist;
data.Dtraining = DtrueTraining;
data.DtestTraining = DtrueTestTraining;
