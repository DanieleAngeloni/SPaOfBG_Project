%% generate_dataset_ideal.m
% Ideal dataset for Robust PCA (matched RPCA assumptions)

clear; clc; close all;
addpath('utils');
rng(0);

%% ============================================================
% Parameters
% ============================================================

N = 100;
T = 500;
r = 5;

num_samples = 120;
num_train   = 80;
num_test    = num_samples - num_train;

L_scale = 10;
sparsity_level   = 0.05;
S_to_L_ratio     = 0.30;
noise_to_L_ratio = 0.05;

fprintf('Generating IDEAL dataset...\n');

%% ============================================================
% Allocate
% ============================================================

Y_all  = zeros(N,T,num_samples);
L0_all = zeros(N,T,num_samples);
S0_all = zeros(N,T,num_samples);

%% ============================================================
% Generation loop
% ============================================================

for i = 1:num_samples

    % Low-rank component
    A = randn(N,r);
    B = randn(T,r);
    L0 = A * B';
    L0 = L0 / norm(L0,'fro') * L_scale;

    % Sparse component (uniform random support)
    S0 = zeros(N,T);
    num_entries = round(sparsity_level * N * T);
    idx = randperm(N*T, num_entries);
    S0(idx) = randn(num_entries,1);
    S0 = S0 / norm(S0,'fro') * (S_to_L_ratio * norm(L0,'fro'));

    % Noise
    noise = randn(N,T);
    noise = noise / norm(noise,'fro');
    noise = noise_to_L_ratio * norm(L0,'fro') * noise;

    % Observation
    Y = L0 + S0 + noise;

    Y_all(:,:,i)  = Y;
    L0_all(:,:,i) = L0;
    S0_all(:,:,i) = S0;
end

fprintf('Generation completed.\n');

%% ============================================================
% Train / Test split
% ============================================================

Y_train  = Y_all(:,:,1:num_train);
L0_train = L0_all(:,:,1:num_train);
S0_train = S0_all(:,:,1:num_train);

Y_test  = Y_all(:,:,num_train+1:end);
L0_test = L0_all(:,:,num_train+1:end);
S0_test = S0_all(:,:,num_train+1:end);

%% ============================================================
% Save dataset
% ============================================================

if ~exist('data','dir'), mkdir('data'); end

save('data/traffic_dataset_theoretical.mat', ...
     'Y_train','L0_train','S0_train', ...
     'Y_test','L0_test','S0_test', ...
     'N','T','r','num_samples', ...
     'num_train','num_test', ...
     '-v7.3');

fprintf('✓ IDEAL dataset saved successfully\n');

%% ============================================================
% Validation + Visualization (shared utility)
% ============================================================

validate_and_visualize_dataset( ...
    Y_all, L0_all, S0_all, ...
    N, T, num_samples, ...
    'ideal');

fprintf('\n========================================\n');
fprintf('IDEAL DATASET GENERATION COMPLETED\n');
fprintf('========================================\n');
