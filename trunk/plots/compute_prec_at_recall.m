function precs = compute_prec_at_recall(r, p, recs)

n = numel(p);
if (n ~= numel(r))
  error('two first arguments should be of the same size');
end

for j=1:numel(recs)
  rec = recs(j);
  done = 0;
  for i=1:n-1
    if (r(i) <= rec && rec <= r(i+1))
      done = 1;
      if (r(i) == r(i+1))
	precs(j) = (p(i) + p(i+1))/2;
      else
	precs(j) = p(i) + (rec - r(i)) * (p(i+1)-p(i))/(r(i+1)-r(i));
      end
      break;
    end
  end
  
  if ~done
    error('not done for %.2f!', rec);
  end
end


