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
x1 = -5:0.1:5; f = x1.^3 + x1.^2 - 1;
fnoisy = f + randn(size(f)) * 20;
p1 = con2seq(x1); t1 = con2seq(fnoisy);

% high resolution for plotting underlying function
xx1 = -5:0.01:5; ff = xx1.^3 + xx1.^2 - 1;

% Setting-up network
%%%%%%%%%%%%%%%%%%%%

% L-M
netf1_4 = feedforwardnet(10,'trainlm');
netf1_4.trainParam.epochs=4;
netf1_4 = train(netf1_4,p1,t1);
a11_4 = sim(netf1_4,p1);

netf1_1000 = feedforwardnet(10,'trainlm');
netf1_1000.trainParam.epochs=1000;
netf1_1000 = train(netf1_1000,p1,t1);
a11_1000 = sim(netf1_1000,p1);

% traingd
netf1_20 = feedforwardnet(10,'traincgf');
netf1_20.trainParam.epochs=20;
netf1_20 = train(netf1_20,p1,t1);
a21_20 = sim(netf1_20,p1);

netf1_1000 = feedforwardnet(10,'traincgf');
netf1_1000.trainParam.epochs=1000;
netf1_1000 = train(netf1_1000,p1,t1);
a21_1000 = sim(netf1_1000,p1);

% Plotting results
%%%%%%%%%%%%%%%%%%

figure;
figure('Color',[1 1 1]);
subplot(2,1,1);
plot(xx1, ff, 'b', x1, fnoisy, 'ko', x1, cell2mat(a11_4), '--g', x1, cell2mat(a11_1000), '--r');
h_legend = legend('true target', 'noise', 'trainlm 20', 'trainlm 1000'); 
set(h_legend,'FontSize',14);
subplot(2,1,2);
plot(xx1, ff, 'b', x1, fnoisy, 'ko', x1, cell2mat(a21_20), '--g', x1, cell2mat(a21_1000), '--r');
h_legend = legend('true target', 'noise', 'traingcf 20', 'traingcf 1000'); 
set(h_legend,'FontSize',14);
export_fig('trainnoise.pdf')
