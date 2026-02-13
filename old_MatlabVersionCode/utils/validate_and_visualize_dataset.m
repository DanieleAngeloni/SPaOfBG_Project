function validate_and_visualize_dataset( ...
    Y_all, L0_all, S0_all, ...
    N, T, num_samples, ...
    dataset_name)

% ------------------------------------------------------------
% Validation
% ------------------------------------------------------------

ranks    = zeros(num_samples,1);
snr_vals = zeros(num_samples,1);
ratio_SL = zeros(num_samples,1);
sparsity = zeros(num_samples,1);

for i = 1:num_samples

    s = svd(L0_all(:,:,i));
    energy = cumsum(s.^2)/sum(s.^2);
    ranks(i) = find(energy > 0.95,1);

    signal = norm(L0_all(:,:,i),'fro')^2 + ...
             norm(S0_all(:,:,i),'fro')^2;

    noise_power = norm(Y_all(:,:,i) - ...
                       L0_all(:,:,i) - ...
                       S0_all(:,:,i),'fro')^2;

    snr_vals(i) = 10*log10(signal/noise_power);

    ratio_SL(i) = norm(S0_all(:,:,i),'fro') / ...
                  norm(L0_all(:,:,i),'fro');

    sparsity(i) = nnz(abs(S0_all(:,:,i)) > 1e-6)/(N*T);
end

fprintf('\nValidation summary (%s)\n', dataset_name);
fprintf('Effective rank: %.1f ± %.1f\n', mean(ranks), std(ranks));
fprintf('Sparsity: %.1f%% ± %.1f%%\n', ...
        mean(sparsity)*100, std(sparsity)*100);
fprintf('SNR: %.1f ± %.1f dB\n', ...
        mean(snr_vals), std(snr_vals));
fprintf('||S||/||L||: %.3f ± %.3f\n', ...
        mean(ratio_SL), std(ratio_SL));

% ------------------------------------------------------------
% Visualization
% ------------------------------------------------------------

if ~exist('figures','dir'), mkdir('figures'); end

sample = 1;

figure('Position',[100,100,1400,800]);

subplot(2,3,1);
imagesc(L0_all(:,:,sample)); colorbar;
title('L_0'); xlabel('Time'); ylabel('Node');
colormap(gca,'jet');

subplot(2,3,2);
imagesc(S0_all(:,:,sample)); colorbar;
title('S_0'); xlabel('Time');
colormap(gca,'jet');

subplot(2,3,3);
imagesc(Y_all(:,:,sample)); colorbar;
title('Y'); xlabel('Time');
colormap(gca,'jet');

subplot(2,3,4);
s = svd(L0_all(:,:,sample));
semilogy(s/s(1),'o-','LineWidth',2);
grid on;
title('Singular values');
xlabel('Index'); ylabel('σ_i / σ_1');

subplot(2,3,5);
vals = abs(S0_all(:,:,sample));
vals = vals(vals > 1e-6);
if ~isempty(vals)
    histogram(vals,50);
    title('|S_{ij}| distribution');
    grid on;
end

subplot(2,3,6);
activity = sum(abs(S0_all(:,:,sample)) > 1e-6,1);
plot(activity,'LineWidth',2);
grid on;
title('Sparse activity over time');
xlabel('Time');

sgtitle(sprintf('%s dataset sample', dataset_name), ...
        'FontSize',14,'FontWeight','bold');

saveas(gcf, sprintf('figures/%s_sample.png', dataset_name));

% ------------------------------------------------------------
% Statistics summary figure
% ------------------------------------------------------------

figure('Position',[150,150,1200,600]);

subplot(2,2,1);
histogram(ranks,'BinMethod','integers');
title('Effective rank'); grid on;

subplot(2,2,2);
histogram(sparsity*100,20);
title('Sparsity (%)'); grid on;

subplot(2,2,3);
histogram(snr_vals,20);
title('SNR (dB)'); grid on;

subplot(2,2,4);
histogram(ratio_SL,20);
title('||S||/||L||'); grid on;

sgtitle(sprintf('%s dataset statistics', dataset_name), ...
        'FontSize',14,'FontWeight','bold');

saveas(gcf, sprintf('figures/%s_statistics.png', dataset_name));

fprintf('✓ Validation and visualization completed (%s)\n\n', dataset_name);
end
