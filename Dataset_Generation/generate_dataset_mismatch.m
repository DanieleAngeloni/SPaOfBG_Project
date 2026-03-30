%% generate_dataset_mismatch.m
% Mismatch dataset for Robust PCA (violates ideal RPCA assumptions)

clear; clc; close all;
addpath('utils');
rng(1);

%% ============================================================
% Parameters
% ============================================================

N = 100;
T = 500;
r_target = 6;

num_samples = 120;
num_train   = 80;
num_test    = num_samples - num_train;

L_scale = 10.0;
S_to_L_ratio = 0.30;
noise_to_L_ratio = 0.20;
% noise_to_L_ratio = 0.05; % for generating the dataset with the same noise level as the ideal dataset
local_energy_fraction = 0.40;

% Structured anomalies
num_anomaly_groups = 4;
nodes_per_group    = 6;
min_duration = 80;
max_duration = 160;
anomaly_amplitude = 0.8;

ar_coeff = 0.95;

fprintf('Generating MISMATCH dataset...\n');

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

    % ---------------- Global low-rank (time-varying subspace) ----------------

    L_global = zeros(N,T);
    split_point = round(T/2);
    segments = {1:split_point, split_point+1:T};

    for seg = 1:2
        idx = segments{seg};
        Tseg = numel(idx);

        A = randn(N, r_target);
        B = zeros(Tseg, r_target);

        for k = 1:r_target
            B(:,k) = filter(1, [1 -ar_coeff], randn(Tseg,1));
        end

        L_seg = A * B';
        L_seg = L_seg / norm(L_seg,'fro');
        L_global(:,idx) = L_seg;
    end

    % ---------------- Local low-rank perturbation ----------------

    local_rank = 4;
    num_local_nodes = round(0.25 * N);
    local_nodes = randperm(N, num_local_nodes);

    C = randn(num_local_nodes, local_rank);
    D = filter(1, [1 -0.9], randn(T, local_rank));

    L_local = zeros(N,T);
    L_local(local_nodes,:) = C * D';
    L_local = L_local / norm(L_local,'fro');

    alpha = sqrt(local_energy_fraction / (1 - local_energy_fraction));
    L0 = L_global + alpha * L_local;
    L0 = L0 / norm(L0,'fro') * L_scale;

    % ---------------- Structured anomalies ----------------

    S0 = zeros(N,T);
    occupied = false(N,T);
    groups = 0;

    while groups < num_anomaly_groups

        nodes = randperm(N, nodes_per_group);
        duration = randi([min_duration, max_duration]);
        t_start = randi([1, T-duration+1]);
        t_end = t_start + duration - 1;

        if ~any(occupied(nodes, t_start:t_end), 'all')

            occupied(nodes, t_start:t_end) = true;

            base = anomaly_amplitude * ...
                   filter(1, [1 -0.8], randn(1,duration));

            for n = nodes
                S0(n,t_start:t_end) = base + 0.05*randn(1,duration);
            end

            groups = groups + 1;
        end
    end

    S0 = S0 / (norm(S0,'fro') + eps) * ...
         (S_to_L_ratio * norm(L0,'fro'));

    % ---------------- Observation ----------------

    noise = randn(N,T);
    noise = noise / norm(noise,'fro');
    noise = noise_to_L_ratio * norm(L0,'fro') * noise;

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

save('data/traffic_dataset_mismatch.mat', ...
    'Y_train','L0_train','S0_train', ...
    'Y_test','L0_test','S0_test', ...
    'N','T','r_target','num_samples', ...
    'num_train','num_test', ...
    '-v7.3');

fprintf('✓ MISMATCH dataset saved successfully\n');

%% ============================================================
% Validation + Visualization (shared utility)
% ============================================================

validate_and_visualize_dataset( ...
    Y_all, L0_all, S0_all, ...
    N, T, num_samples, ...
    'mismatch');

fprintf('\n========================================\n');
fprintf('MISMATCH DATASET GENERATION COMPLETED\n');
fprintf('========================================\n');
