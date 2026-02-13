%% run1_admm_ideal.m
% RUN1 — ADMM (Ideal)
% Classical model-based ADMM on theoretical dataset

clear; clc; close all;
addpath('utils');

if ~exist('results','dir'), mkdir('results'); end

%% Load dataset

load('data/traffic_dataset_theoretical_full.mat');

fprintf('Theoretical dataset loaded\n');
fprintf('Train: %d | Test: %d\n', num_train, num_test);

[N,T] = size(Y_test(:,:,1));

%% ADMM parameters (theoretical choice)

lambda = 1 / sqrt(max(N,T));
rho = 1.0;

max_iter = 2000;
tol = 1e-4;

fprintf('lambda = %.2e | rho = %.2f\n', lambda, rho);

%% Containers

NMSE_L = zeros(num_test,1);
NMSE_S = zeros(num_test,1);
Iterations = zeros(num_test,1);

fprintf('\nRunning ADMM on theoretical test set...\n');

%% ADMM loop

for ii = 1:num_test

    Y  = Y_test(:,:,ii);
    L0 = L0_test(:,:,ii);
    S0 = S0_test(:,:,ii);

    L = zeros(N,T);
    S = zeros(N,T);
    U = zeros(N,T);

    for k = 1:max_iter

        % L update (SVT)
        [U_svd,Sigma,V_svd] = svd(Y - S + U,'econ');
        s = max(diag(Sigma) - 1/rho, 0);
        L_new = U_svd * diag(s) * V_svd';

        % S update
        S_new = soft_threshold(Y - L_new + U, lambda / rho);

        % Dual update
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

    if ii == 1 || mod(ii,10)==0
        fprintf('Test %3d/%3d | NMSE_L=%.4f | NMSE_S=%.4f | Iter=%d\n', ...
            ii, num_test, NMSE_L(ii), NMSE_S(ii), k);
    end
end

%% Results

fprintf('\n===== RUN1 — ADMM (Ideal) =====\n');
fprintf('NMSE_L     : %.6f ± %.6f\n', mean(NMSE_L), std(NMSE_L));
fprintf('NMSE_S     : %.6f ± %.6f\n', mean(NMSE_S), std(NMSE_S));
fprintf('Iterations : %.1f ± %.1f\n', mean(Iterations), std(Iterations));
fprintf('================================\n');

%% Save

save('results/RUN1_ADMM_IDEAL.mat', ...
     'NMSE_L','NMSE_S','Iterations','lambda','rho');

fprintf('\nRUN1 completed.\n');
