function Y = soft_threshold(X, tau)
Y = sign(X) .* max(abs(X) - tau, 0);
end
