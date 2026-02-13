%% ============================================================
%  run3_ADMM_global_learned.m
%
%  RUN3 — Global-Learned ADMM
%
%  - Same ADMM model as baseline
%  - (lambda, rho) selected via offline grid search
%  - Selection criterion: sqrt(NMSE_L * NMSE_S)
% ============================================================

clear; clc; close all;
addpath('utils');

if ~exist('results','dir'), mkdir('results'); end

% Load dataset
load('data/traffic_dataset_mismatch.mat');

fprintf('Mismatch dataset loaded\n');
fprintf('Train: %d | Test: %d\n', num_train, num_test);

[N,T,~] = size(Y_train);

% ADMM settings
max_iter_train = 300;
tol_train      = 1e-3;

max_iter_test  = 2000;
tol_test       = 1e-5;

% GRID SEARCH (TRAIN)
fprintf('\n=== GRID SEARCH (TRAIN) ===\n');

lambda0 = 1 / sqrt(max(N,T));

lambda_grid = lambda0 * [0.25 0.5 0.75 1 1.25 1.5 2];
rho_grid    = [0.5 0.75 1.0 1.25 1.5 2.0];

train_subset = 1:min(25, num_train);

best_score  = inf;
lambda_best = NaN;
rho_best    = NaN;

for lambda = lambda_grid
    for rho = rho_grid

        scores = zeros(numel(train_subset),1);

        for jj = 1:numel(train_subset)

            idx = train_subset(jj);

            Y  = Y_train(:,:,idx);
            L0 = L0_train(:,:,idx);
            S0 = S0_train(:,:,idx);

            [n,t] = size(Y);

            L = zeros(n,t);
            S = zeros(n,t);
            U = zeros(n,t);

            for k = 1:max_iter_train

                [U_svd,Sigma,V_svd] = svd(Y - S + U,'econ');
                s = max(diag(Sigma) - 1/rho, 0);
                L_new = U_svd * diag(s) * V_svd';

                S_new = soft_threshold(Y - L_new + U, lambda / rho);

                residual = Y - L_new - S_new;
                U = U + residual;

                if norm(residual,'fro') / norm(Y,'fro') < tol_train
                    L = L_new;
                    S = S_new;
                    break;
                end

                L = L_new;
                S = S_new;
            end

            nmse_L = norm(L - L0,'fro')^2 / norm(L0,'fro')^2;
            nmse_S = norm(S - S0,'fro')^2 / norm(S0,'fro')^2;

            scores(jj) = sqrt(nmse_L * nmse_S);
        end

        mean_score = mean(scores);

        fprintf('lambda=%.4f | rho=%.2f | score=%.4f\n', ...
                lambda, rho, mean_score);

        if mean_score < best_score
            best_score  = mean_score;
            lambda_best = lambda;
            rho_best    = rho;
        end
    end
end

fprintf('\nBest (train): lambda=%.4f | rho=%.2f | score=%.4f\n', ...
        lambda_best, rho_best, best_score);

% TEST (FULL SET)
fprintf('\n=== TESTING WITH LEARNED PARAMETERS ===\n');

NMSE_L     = zeros(num_test,1);
NMSE_S     = zeros(num_test,1);
Iterations = zeros(num_test,1);

for ii = 1:num_test

    Y  = Y_test(:,:,ii);
    L0 = L0_test(:,:,ii);
    S0 = S0_test(:,:,ii);

    [n,t] = size(Y);

    L = zeros(n,t);
    S = zeros(n,t);
    U = zeros(n,t);

    for k = 1:max_iter_test

        [U_svd,Sigma,V_svd] = svd(Y - S + U,'econ');
        s = max(diag(Sigma) - 1/rho_best, 0);
        L_new = U_svd * diag(s) * V_svd';

        S_new = soft_threshold(Y - L_new + U, lambda_best / rho_best);

        residual = Y - L_new - S_new;
        U = U + residual;

        if norm(residual,'fro') / norm(Y,'fro') < tol_test
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

% Aggregate results
fprintf('\n===== RUN3 — GLOBAL-LEARNED ADMM =====\n');
fprintf('NMSE_L     : %.6f ± %.6f\n', mean(NMSE_L), std(NMSE_L));
fprintf('NMSE_S     : %.6f ± %.6f\n', mean(NMSE_S), std(NMSE_S));
fprintf('Iterations : %.1f ± %.1f\n', mean(Iterations), std(Iterations));
fprintf('=======================================\n');

% Save
save('results/results_RUN3_GLOBAL_LEARNED_ADMM.mat', ...
     'NMSE_L','NMSE_S','Iterations', ...
     'lambda_best','rho_best');

fprintf('\nRUN3 completed.\n');
