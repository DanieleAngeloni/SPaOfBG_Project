function plot_comparison(L0, L_hat, S0, S_hat, metrics, method_name, filename)

figure('Position',[100 100 1200 500]);

subplot(2,3,1);
imagesc(L0); colorbar;
title('L_0 (ground truth)','Interpreter','none');

subplot(2,3,2);
imagesc(L_hat); colorbar;
title('L̂','Interpreter','none');

subplot(2,3,4);
imagesc(S0); colorbar;
title('S_0 (ground truth)','Interpreter','none');

subplot(2,3,5);
imagesc(S_hat); colorbar;
title('Ŝ','Interpreter','none');

if ~isempty(metrics)
    subplot(2,3,[3 6]);
    semilogy(metrics.err_history,'LineWidth',2);
    grid on;
    xlabel('Iteration / Layer');
    ylabel('||Y - L - S||_F / ||Y||_F');
    title('Convergence','Interpreter','none');
end

sgtitle(sprintf('%s | NMSE_L=%.3f  NMSE_S=%.3f  F1=%.2f  Iter=%d', ...
    method_name, ...
    metrics.NMSE_L, metrics.NMSE_S, ...
    metrics.F1, metrics.num_iter), ...
    'Interpreter','none');

saveas(gcf, filename);
close;

end
