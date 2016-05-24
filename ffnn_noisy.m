% Approximation of non linear function + gaussian noise using feedforward neural
% networks with various training algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
close all

addpath export_fig

% Setting up non linear function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions to learn
x1 = -5:0.05:5; f = x1.^3 + x1.^2 - 1;

fnoisy = f + randn(size(f)) * 10;
p1 = con2seq(x1); t1 = con2seq(fnoisy);

% high resolution for plotting underlying function
xx1 = -5:0.01:5; ff = xx1.^3 + xx1.^2 - 1;

figure('Color',[1 1 1]);
subplot(2,2,2);
plot(xx1, ff, 'b', x1, fnoisy, 'ro');
subplot(2,2,2);
title('Function approximation','FontSize',18,'FontWeight', 'normal');
xlabel('X','FontSize',14); ylabel('Y','FontSize',14);
h_legend = legend('real function','noisy dataset');
set(h_legend,'FontSize',18);

% Setting up training set and validation set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [trainInd, testInd] = dividerand(200,0.7,0.3);
% xt = x1(trainInd); yt = fnoisy(trainInd);
% x_test = x1(testInd); y_test = fnoisy(testInd);

%%% test classifier after 1000 epochs
training_algorithm=char('traingda','trainlm','trainbfg','traincgf');
colors = {'b-.', 'ro', 'm-.', 'c-.'};
i = 1;

count = 1;
for alg = 1:4
    algorithm = char(strcat(training_algorithm(alg,:)));

    net=feedforwardnet(10, char(algorithm));
    init(net);
    net.trainParam.epochs=1000;
    net.trainParam.showWindow = false;
    [net,tr]=train(net,p1,t1);
    a = sim(net,p1);

    subplot(1,2,1);
    hold on;
    plot(x1,cell2mat(a),char(colors(i)));
    
    count = count + 1;
    i = i + 1;
end

title('Function approximation','FontSize',18,'FontWeight', 'normal');
xlabel('X','FontSize',14); ylabel('Y','FontSize',14);
h_legend = legend('traingda', 'trainlm', 'trainbfg', 'traincgf','FontSize',18);
set(h_legend,'FontSize',14);

% Main loop
%%%%%%%%%%%
training_algorithm=char('traingda','trainlm','trainbfg','traincgf');
epoch_list = linspace(100, 1000, 10);
err_matrix = zeros(4,10); i = 1; j = 1;
colors = {'b-.', 'ro', 'm-.', 'c-.'};

for alg = 1:4
    count = 1; j = 1;
    algorithm = char(strcat(training_algorithm(alg,:)));
    for ep = epoch_list
        % build and train ffnnet
        net=feedforwardnet(10, char(algorithm));
        init(net);
        net.trainParam.epochs=ep;
        net.trainParam.showWindow = false;
        [net,tr]=train(net,p1,t1);
        a = sim(net,p1);
        err_matrix(i, j) = sum((cell2mat(a) - f).^2); 
        j = j + 1;
    end        

    subplot(2,2,4);
    hold on;
    semilogy(epoch_list, err_matrix(i,:),char(colors(i)));
    
    count = count + 1;
    i = i + 1;
end

subplot(2,2,4);
title('Training epochs','FontSize',18, 'FontWeight', 'normal');
xlabel('Epochs','FontSize',14); ylabel('True MSE','FontSize',14);
g_legend = legend('traingd', 'trainlm', 'trainbfg', 'traincgf','FontSize',18);
set(g_legend,'FontSize',18);

export_fig('ffnn_noisy_testmse.pdf');

