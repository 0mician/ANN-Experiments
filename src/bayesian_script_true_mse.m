clc, clear all, close all;
addpath 'export_fig';

x = -pi:0.1:pi; y = sin(x); yy = y;
fn = y + randn(size(y))*0.2;
p= con2seq(x); t = con2seq(fn); 

neurons_list = 1:30;
nepochs = 100;
ndatasets = 25;
algorithms = char('traingd','trainbfg','traincgf','trainlm','trainbr');
perf_true_matrix = zeros(5, length(neurons_list));

figure('Color', [1 1 1]);
title('Performance for various training algorithm','FontSize',18, 'FontWeight', 'normal');
xlabel('Neurons','FontSize',16); ylabel('MSE','FontSize',16);

tic;
for algorithm = 1:size(algorithms, 1)
    rng(1);
    algorithm_name = char(strcat(algorithms(algorithm,:)));

    for nn = neurons_list
        disp(sprintf('Learning with %s: neural net consisting of %i neurons...', algorithm_name, nn));
        perf_true = zeros(size(ndatasets));
        parfor dataset = 1:ndatasets            
            fn = y + randn(size(y))*0.2; % new dataset to train
            t = con2seq(fn);
            
            net=feedforwardnet(nn, algorithm_name);
            net.trainParam.epochs = nepochs;  
            net.trainParam.showWindow = false;
            net=train(net,p,t); 

            perf_true(dataset) = perform(net, y, net(x));
        end
        perf_true_matrix(algorithm, nn) = sum(perf_true) / ndatasets;
    end
    hold on;
    plot(neurons_list,perf_true_matrix(algorithm, :), 'LineWidth', 2)
end
toc;

hold on;
set(gca,'yscale','log');
legend('traingd','trainbfg','traincgf','trainlm','trainbr');
grid on;

export_fig('true_mse_perf.pdf');
