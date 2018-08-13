function A = auroc(tp, fp)

n = size(tp, 1);
A = sum((fp(2:n) - fp(1:n-1)).*(tp(2:n)+tp(1:n-1)))/2;

