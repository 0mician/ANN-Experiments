% Approximation of (non-noisy) non linear function using feedforward neural
clear; clc; close all

addpath export_fig

% Setting up non linear function
x = -pi:0.1:pi ; f = exp(-x.^2).*sin(10.*x);
p = con2seq(x); t = con2seq(f);

% high resolution for plotting underlying function
xx = -pi:0.01:pi ; ff = exp(-xx.^2).*sin(10.*xx);

% Available training algorithms 4 selected: gd, levenberg-maquardt, 
% quasi newton, conjugate gradient
training_algorithm=char('traingd','trainlm','trainbfg','traincgf');
colors = {'b-.', 'ro', 'm-.', 'c-.'};

% Plotting 
figure('Color',[1 1 1]);
fun_graph = subplot(1,2,1);
plot(xx, ff,'k'); % underlying real function
epoch_graph = subplot(1,2,2);

% Training (4 training algorithm)
for i = 1:4
    
    algorithm = char(strcat(training_algorithm(i,:)));
    
    % build and train ffnnet
    net=feedforwardnet(10, char(algorithm));
    init(net);
    net.trainParam.epochs=100;
    net.trainParam.showWindow = false;
    [net,tr]=train(net,p,t);
    a = sim(net,p);
    
    perf = tr.perf;
    size(perf)
    epochs = tr.epoch;
    size(epochs)
    
    % plots (approx and perf)
    subplot(1,2,1);
    hold on;
    plot(x,cell2mat(a),char(colors(i)));

    subplot(1,2,2);
    loglog(epochs,perf, char(colors(i)))
    hold on;
end

% Polishing of plots
subplot(1,2,1);
set(get(fun_graph,'Title'),'String', 'Function approximation','FontSize',18,'FontWeight', 'normal');
set(get(fun_graph,'Xlabel'),'String', 'X','FontSize',14)
set(get(fun_graph,'Ylabel'),'String', 'Y','FontSize',14)
h_legend = legend('real function','traingd', 'trainlm', 'trainbfg', 'traincgf','FontSize',18);
set(h_legend,'FontSize',14);

subplot(1,2,2);
set(get(epoch_graph,'Title'),'String', 'Training epochs','FontSize',18, 'FontWeight', 'normal');
set(get(epoch_graph,'Xlabel'),'String', 'Epoch (log scale)','FontSize',14)
set(get(epoch_graph,'Ylabel'),'String', 'Training MSE (log scale)','FontSize',14)
g_legend = legend('traingd', 'trainlm', 'trainbfg', 'traincgf','FontSize',18);
set(g_legend,'FontSize',14);
set(gca,'yscale','log');
set(gca,'xscale','log');

export_fig('training_performance.pdf')
