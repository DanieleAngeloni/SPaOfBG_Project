%% run2_admm_mismatch.m
% RUN2 — ADMM (Baseline Mismatch)
% Classical model-based ADMM under model mismatch

clear; clc; close all;
addpath('utils');

if ~exist('results','dir'), mkdir('results'); end
if ~exist('figures','dir'), mkdir('figures'); end

%% Load dataset

load('data/traffic_dataset_mismatch.mat');

fprintf('Mismatch dataset loaded\n');
fprintf('Train: %d | Test: %d\n', num_train, num_test);

[N,T] = size(Y_test(:,:,1));

%% ADMM parameters

lambda = 1 / sqrt(max(N,T));
rho = 1.0;

max_iter = 2000;
tol = 1e-5;

fprintf('lambda = %.2e | rho = %.2f\n', lambda, rho);

%% Containers

NMSE_L = zeros(num_test,1);
NMSE_S = zeros(num_test,1);
Iterations = zeros(num_test,1);

L_rep = [];
S_rep = [];
L0_rep = [];
S0_rep = [];

fprintf('\nRunning ADMM on mismatch test set...\n');

%% ADMM loop

for ii = 1:num_test

    Y  = Y_test(:,:,ii);
    L0 = L0_test(:,:,ii);
    S0 = S0_test(:,:,ii);

    L = zeros(N,T);
    S = zeros(N,T);
    U = zeros(N,T);

    for k = 1:max_iter

        % L update
        [U_svd,Sigma,V_svd] = svd(Y - S + U,'econ');
        s = max(diag(Sigma) - 1/rho, 0);
        L_new = U_svd * diag(s) * V_svd';

        % S update
        S_new = soft_threshold(Y - L_new + U, lambda / rho);

        residual = Y - L_new - S_new;
        U = U + residual;

        if norm(residual,'fro') / norm(Y,'fro') < tol
            L = L_new;
            S = S_new;
            break;
        end

        L = L_new;
        S = S_new;
    end

    NMSE_L(ii) = norm(L - L0,'fro')^2 / norm(L0,'fro')^2;
    NMSE_S(ii) = norm(S - S0,'fro')^2 / norm(S0,'fro')^2;
    Iterations(ii) = k;

    if ii == 1
        L_rep = L;
        S_rep = S;
        L0_rep = L0;
        S0_rep = S0;
    end

    if ii == 1 || mod(ii,10)==0
        fprintf('Test %3d/%3d | NMSE_L=%.4f | NMSE_S=%.4f | Iter=%d\n', ...
            ii, num_test, NMSE_L(ii), NMSE_S(ii), k);
    end
end

%% Results

fprintf('\n===== RUN2 — ADMM (Baseline Mismatch) =====\n');
fprintf('NMSE_L     : %.6f ± %.6f\n', mean(NMSE_L), std(NMSE_L));
fprintf('NMSE_S     : %.6f ± %.6f\n', mean(NMSE_S), std(NMSE_S));
fprintf('Iterations : %.1f ± %.1f\n', mean(Iterations), std(Iterations));
fprintf('============================================\n');

%% Save

save('results/RUN2_ADMM_BASELINE_MISMATCH.mat', ...
     'NMSE_L','NMSE_S','Iterations','lambda','rho');

%% Representative visualization (first test sample)

figure('Position',[100 100 1200 600]);

subplot(1,3,1);
imagesc(L0_rep); colorbar;
title('True L_0'); xlabel('Time'); ylabel('Node');

subplot(1,3,2);
imagesc(L_rep); colorbar;
title('Estimated L'); xlabel('Time');

subplot(1,3,3);
imagesc(abs(S_rep)); colorbar;
title('Estimated |S|'); xlabel('Time');

sgtitle('RUN2 — ADMM (Baseline Mismatch)');

saveas(gcf, 'figures/RUN2_ADMM_MISMATCH_rep.png');

fprintf('\nRUN2 completed.\n');
