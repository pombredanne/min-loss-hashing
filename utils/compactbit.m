function cb = compactbit(b, wordsize)
% cb = compacted string of bits (using words of 'wordsize' bits)

if (~exist('wordsize'))
  wordsize = 8;
end

[nSamples nbits] = size(b);
nwords = ceil(nbits/wordsize);

if (wordsize == 8)
  cb = zeros([nSamples nwords], 'uint8');
elseif (wordsize == 16)
  cb = zeros([nSamples nwords], 'uint16');
elseif (wordsize == 32)
  cb = zeros([nSamples nwords], 'uint32');
elseif (wordsize == 64)
  cb = zeros([nSamples nwords], 'uint64');
else
  error('unrecognized wordsize');
end

for j = 1:nbits
    w = ceil(j/wordsize);
    cb(:,w) = bitset(cb(:,w), mod(j-1,wordsize)+1, b(:,j));
end
