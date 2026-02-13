function metrics = compute_metrics(L_hat, S_hat, L0, S0, err_history, threshold)

% ===============================
% Reconstruction errors
% ===============================
metrics.NMSE_L = norm(L_hat - L0, 'fro')^2 / norm(L0, 'fro')^2;
metrics.NMSE_S = norm(S_hat - S0, 'fro')^2 / max(norm(S0, 'fro')^2, 1e-12);

% ===============================
% Anomaly detection metrics
% ===============================
S_detect = abs(S_hat) > threshold;
S_true   = abs(S0) > 0;

TP = sum(S_detect(:) & S_true(:));
FP = sum(S_detect(:) & ~S_true(:));
FN = sum(~S_detect(:) & S_true(:));

metrics.precision = TP / (TP + FP + eps);
metrics.recall    = TP / (TP + FN + eps);
metrics.F1        = 2 * metrics.precision * metrics.recall / ...
                    (metrics.precision + metrics.recall + eps);

% ===============================
% Convergence information
% ===============================
metrics.err_history = err_history;
metrics.num_iter    = length(err_history);

end
