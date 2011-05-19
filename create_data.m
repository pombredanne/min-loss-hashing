function data = create_data(MODE, operand1, operand2, operand3)

data.MODE = MODE;

if (strcmp(MODE, 'uniform'))
  % Create uniformly distributed synthesize data:

  % parameters
  Ntraining = 3000; % number training samples
  Ntest = 3000;     % number test samples
  averageNumberNeighbors = 50; % number of ground-truth neighbors for each training point (on average)
  aspectratio = ones(1, dtr);  % aspect ratio of different edge lenghts of the uniform hypercube

  % uniform distribution
  dtr = operand1;

  Xtraining = rand([Ntraining, dtr]);
  for i=1:dtr
    Xtraining(:,i) = aspectratio(i) * Xtraining(:,i) - aspectratio(i)/2;
  end
  Xtest = rand([Ntest,dtr]);
  for i=1:dtr
    Xtest(:,i) = aspectratio(i)*Xtest(:,i) - aspectratio(i)/2;
  end

  data.aspectratio = aspectratio;
  data = construct_data(Xtraining, Xtest, [Ntraining, Ntest], averageNumberNeighbors, data); 
  % see bottom for construct_data(...)
  
elseif (strcmp(MODE, 'euc-22K-labelme'))
  % Create Euclidean 22K labelme dataset:
  load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'gist');

  X = gist;
  clear gist;
  
  X = X - ones(size(X,1),1)*mean(X);
  for i = 1:size(X,1)
    X(i,:) = X(i,:) / norm(X(i,:));
  end
  
  Xtraining = X(ndxtrain, :);
  Xtest = X(ndxtest, :);
  
  data = construct_data(Xtraining, Xtest, [numel(ndxtrain), numel(ndxtest)], operand1, data);
  % see bottom for construct_data(...)
  
elseif (strcmp(MODE, 'sem-22K-labelme'))
  % Create semantic 22K labelme dataset:
  load('data/LabelMe_gist', 'ndxtrain', 'ndxtest', 'DistLM', 'gist');

  X = gist;
  clear gist;
  
  X = X - ones(size(X,1),1)*mean(X);
  for i = 1:size(X,1)
    X(i,:) = X(i,:) / norm(X(i,:));
  end
  if (~exist('operand3'))
    scale = 1;
  else
    scale = operand3;
  end
  X = X * scale;
  data.scale = scale;

  Xtraining = X(ndxtrain, :);
  Xtest     = X(ndxtest, :);
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

elseif (strcmp(MODE, 'sem-full-mnist') || strcmp(MODE, 'sem-full-mnist2'))
  % Create Semantic full MNIST
  load('data/mnist-full.mat');

  Ntraining = 60000;
  Ntest = 10000;
  WtrueTraining = sparse(false(Ntraining, Ntraining));
  for i = 0:9
    WtrueTraining = WtrueTraining + logical(sparse(double(mnist_ltrain == i)) * sparse(double(mnist_ltrain == i))');
  end
  WtrueTestTraining = sparse(false(Ntest, Ntraining));
  for i = 0:9
    WtrueTestTraining = WtrueTestTraining & logical(sparse(double(mnist_ltest == i)) * sparse(double(mnist_ltrain == i))');
  end
  
  X = double([mnist_train; mnist_test]);
  if (strcmp(MODE, 'sem-full-mnist')) % no mean-centering and normalization by default
    X = X / 255;
  else
    mean_mnist = mean(mnist_train);
    X = X - ones(size(X,1),1)*mean_mnist;
    for i = 1:size(X,1)
      X(i,:) = X(i,:) / norm(X(i,:));
    end
    data.mean_mnist = mean_mnist;
  end

  rperm = randperm(Ntraining);
  data.Ntraining = Ntraining;
  data.Ntest = Ntest;
  data.Ltraining = mnist_ltrain(rperm);
  data.Xtraining = X(rperm(1:60000), :);
  data.WtrueTraining = WtrueTraining(rperm, rperm);
  data.Ltest = mnist_ltest;
  data.Xtest = X(60001:end, :);
  data.WtrueTestTraining = WtrueTestTraining;

elseif (strcmp(MODE, 'kulis'))
  % from Brian Kulis's Code
  data.MODE = [MODE, ' - ', operand1];
  X = load(['data/kulis/', operand1, '.mtx']);

  averageNumberNeighbors = 50; % number of ground-truth neighbors for each training point (on average)
  Ntraining = 1000;
  Ntest = min(3000, size(X,1)-1000);

  % center, then normalize data
  X = X - ones(size(X,1),1)*mean(X);
  for i = 1:size(X,1)
    X(i,:) = X(i,:) / norm(X(i,:));
  end
  
  rp = randperm(size(X,1));
  trIdx = rp(1:Ntraining);
  testIdx = rp(Ntraining+1:Ntraining+Ntest);
  Xtraining = X(trIdx,:);
  Xtest = X(testIdx,:);
  
  data = construct_data(Xtraining, Xtest, [Ntraining, Ntest], averageNumberNeighbors, data);
  % see bottom for construct_data(...)
else
  fprintf('The given mode is not recognized.\n');
end


function data = construct_data(Xtraining, Xtest, sizeSets, averageNumberNeighbors, data)

  [Ntraining, Ntest] = deal(sizeSets(1), sizeSets(2));
  DtrueTraining = distMat(Xtraining);
  fprintf('DtrueTraining is done.\n');
  Dball = sort(DtrueTraining, 2);
  Dballmin = mean(Dball(:,averageNumberNeighbors));

  DtrueTestTraining = distMat(Xtest,Xtraining); % size = [Ntest x Ntraining]
  fprintf('DtrueTestTraining is done.\n');

  [sorted ind] = sort(DtrueTraining(:));
  endTrain = sum(sorted < Dballmin);

  data.Xtraining = Xtraining;
  data.Xtest = Xtest;  
  data.WtrueTraining = DtrueTraining < Dballmin;
  data.WtrueTestTraining = DtrueTestTraining < Dballmin;

  data.Ntraining = Ntraining;
  data.Ntest = Ntest;
  data.averageNumberNeighbors = averageNumberNeighbors;
  data.Dballmin = Dballmin;
  data.Dtraining = DtrueTraining;
  data.DtestTraining = DtrueTestTraining;
