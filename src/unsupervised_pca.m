% naive implementation of PCA
function [evec,reduced,d] = unsupervised_pca(x, n)
a = cov(x);
[evec,eval] = eigs(a, n);
d = diag(eval);
reduced = transpose(evec)*x';
end