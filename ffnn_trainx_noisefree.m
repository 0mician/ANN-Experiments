% Approximation of (non-noisy) non linear function using feedforward neural
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
p1 = con2seq(x1); t1 = con2seq(f);

x2 = -pi:0.1:pi ; g = exp(-x2.^2).*sin(10.*x2);
p2 = con2seq(x2); t2 = con2seq(g);

% high resolution for plotting underlying function
xx1 = -5:0.01:5; ff = xx1.^3 + xx1.^2 - 1;
xx2 = -pi:0.01:pi ; gg = exp(-xx2.^2).*sin(10.*xx2);

% Setting-up network
%%%%%%%%%%%%%%%%%%%%

% Gradient descent
% f(x)
netf1_20 = feedforwardnet(10,'trainlm');
netf1_20.trainParam.epochs=20;
netf1_20 = train(netf1_20,p1,t1);
a11_20 = sim(netf1_20,p1);

netf1_40 = feedforwardnet(10,'trainlm');
netf1_40.trainParam.epochs=40;
netf1_40 = train(netf1_40,p1,t1);
a11_40 = sim(netf1_40,p1);

% g(x)
netf2_20 = feedforwardnet(10,'trainlm');
netf2_20.trainParam.epochs=20;
netf2_20= train(netf2_20,p2,t2);
a21_20 = sim(netf2_20,p2);

netf2_40 = feedforwardnet(10,'trainlm');
netf2_40.trainParam.epochs=40;
netf2_40= train(netf2_40,p2,t2);
a21_40 = sim(netf2_40,p2);
% 
% netf2_1000 = feedforwardnet(10,'traingd');
% netf2_1000.trainParam.epochs=1000;
% netf2_1000 = train(netf2_1000,p2,t2);
% a21_1000 = sim(netf2_1000,p2);

% Plotting
%%%%%%%%%%

figure;
figure('Color',[1 1 1]);
subplot(2,1,1);
plot(xx1,ff,'b',x1,cell2mat(a11_20),'go', x1, cell2mat(a11_40), '--r');
h_legend = legend('true target', 'traingd 20 iterations', 'traingd 40 iterations');
set(h_legend,'FontSize',14);
subplot(2,1,2);
plot(xx2,gg,'b',x2,cell2mat(a21_20),'--g', x2, cell2mat(a21_40), '--r')%, x2, cell2mat(a21_1000), '--k');
h_legend = legend('true target', 'traingd 20 iterations', 'traingd 40 iterations')%, 'traingd 1000 iterations');
set(h_legend,'FontSize',14);

export_fig('trainlm.pdf')