clc, clear all, close all;

load datasets/threes.mat -ascii

% Computing mean, cov, and ev
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Color', [1 1 1]);
subplot(2,2,1);
mean_three = mean(threes, 1);
imagesc(reshape(mean_three,16,16),[0,1]);

cov_matrix = cov(threes);
[ev, d] = eig(cov_matrix);
subplot(1,2,2);
semilogy(diag(d));

subplot(4,6,13);
imagesc(reshape(ev(:,251),16,16),[0,1]);
subplot(4,6,14);
imagesc(reshape(ev(:,252),16,16),[0,1]);
subplot(4,6,15);
imagesc(reshape(ev(:,253),16,16),[0,1]);

subplot(4,6,19);
imagesc(reshape(ev(:,254),16,16),[0,1]);
subplot(4,6,20);
imagesc(reshape(ev(:,255),16,16),[0,1]);
subplot(4,6,21);
imagesc(reshape(ev(:,256),16,16),[0,1]);

% Compression/recon of an image for testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
npca = 4;
figure('Color', [1 1 1]);
i = 1;
for index = 1:2
    subplot(4, 5, i);
    imagesc(reshape(threes(index,:),16,16),[0,1]);
    i = i + 1
    for n=1:npca
        [evec,eval] = eigs(cov_matrix, n);
        projected_image = evec'*threes(index,:)'; projected_image = projected_image';
        recon_image = projected_image*evec';
        subplot(4, 5, i);
        imagesc(reshape(recon_image(1,:),16,16),[0,1]);
        i = i + 1;
    end
end

% Compression of the whole dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = zeros(50, 1);
for npca = 1:50    
    reconstructed_data = zeros(500, 256);
    for index = 1:500
        [evec,eval] = eigs(cov_matrix, npca);
        projected_image = evec'*threes(index,:)'; projected_image = projected_image';
        recon_image = projected_image*evec';
        reconstructed_data(index, :) = recon_image;
    end
    error(npca) = sum(sum((reconstructed_data - threes).^2));
    disp(npca)
end

figure('Color', [1 1 1]);
subplot(2,1,1);
plot(1:50, error, 'b-','linewidth',4);
title('Reconstruction error','FontSize',18,'FontWeight', 'normal');
xlabel('Number of principal components','FontSize',14);
ylabel('Reconstruction error','FontSize',14);


% Reconstruction error with all components == 0?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reconstructed_all = zeros(500, 256);
for index = 1:500
    [evec,eval] = eigs(cov_matrix, 256);
    projected_image = evec'*threes(index,:)'; projected_image = projected_image';
    recon_image = projected_image*evec';
    reconstructed_all(index, :) = recon_image;
end
disp(sum(sum((reconstructed_data - threes).^2)));

% Eigenvalues vs. error.
%%%%%%%%%%%%%%%%%%%%%%%%
[ev, d] = eig(cov_matrix);
deval = diag(d);
deval = sort(deval,'descend');
diff_error = sum(deval) - cumsum(deval(1:50));
subplot(2,1,2);
plot(1:50, diff_error, 'r-','linewidth',4);
title('Eigenvalues contribution','FontSize',18,'FontWeight', 'normal');
xlabel('Number of principal components','FontSize',14);
ylabel('Contribution','FontSize',14);
export_fig('unsupervised_reconstruction_error.pdf');