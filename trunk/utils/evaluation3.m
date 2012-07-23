function [score, recall] = evaluation3(S, Dhamm, maxn)

% Input:
%    S = true neighbors [Ntest * Ndataset], can be a full matrix NxN
%    Dhamm = estimated distances
%    maxn = number of distinct distance values to be considered
%
% Output:
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  score(n) = --------------------------------------------------------------
%               exp. # of total pairs inside hamming ball of radius <= (n-1)
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  recall(n) = --------------------------------------------------------------
%                          exp. # of total good pairs 

if ~islogical(S)
  S = logical(S);
end

[a b] = accumarray_reverse(Dhamm, full(S), maxn+1);
suma = cumsum(sum(a,2));
score = suma ./ cumsum(sum(b,2));
recall = suma' / sum(S(:));
