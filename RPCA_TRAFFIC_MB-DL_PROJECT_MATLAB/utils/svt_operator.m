function L = svt_operator(X, tau)
[U,S,V] = svd(X, 'econ');
s = diag(S);
s = max(s - tau, 0);
L = U * diag(s) * V';
end
