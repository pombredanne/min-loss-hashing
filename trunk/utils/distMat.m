function D = distMat(X1, X2, knn)

% Pairwise Euclidian distances between data points
% Each data point is one column

buffer = 1000;

if nargin >= 2 && ~isempty(X2)
  R = X1'*X2;
  sqrX1 = sum(X1.^2);
  sqrX2 = sum(X2.^2);
  D = bsxfun(@plus, sqrX1', sqrX2);
  D = real(sqrt(D-2*R));
  
  sorted = sort(D,1);
    
  if (exist('knn', 'var'))
    D = sparse(bsxfun(@le, D, sorted(knn,:)));
  end
else
  n = size(X1,2);
  if (~exist('knn', 'var') || n < buffer)
    R = X1'*X1;
    sqrX1 = sum(X1.^2);
    D = bsxfun(@plus, sqrX1', sqrX1);
    D = D - 2*R;
    D = real(sqrt(D));
    
    sorted = sort(D,1);
    
    if (exist('knn', 'var'))
      D = sparse(bsxfun(@le, D, sorted(knn,:)));
    end
    % if (exist('knn', 'var'))
    %   D = sparse(D <= knn);
    % end
  else
    D = sparse(sparse(n,n) > 0);
    for (i=1:ceil(n/buffer))
      fprintf('%d/%d\r', i, ceil(n/buffer));
      D(:, (i-1)*buffer+1:min(i*buffer,n)) = distMat( ...
	  X1, X1(:, (i-1)*buffer+1:min(i*buffer,n)), knn);
    end
  end
end
