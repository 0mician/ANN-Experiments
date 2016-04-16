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
netf1_4 = feedforwardnet(10,'trainlm');
netf1_4.trainParam.epochs=4;
netf1_4 = train(netf1_4,p1,t1);
a11_4 = sim(netf1_4,p1);

netf1_10 = feedforwardnet(10,'trainlm');
netf1_10.trainParam.epochs=10;
netf1_10 = train(netf1_10,p1,t1);
a11_10 = sim(netf1_10,p1);

% g(x)
netf2_40 = feedforwardnet(10,'trainlm');
netf2_40.trainParam.epochs=40;
netf2_40= train(netf2_40,p2,t2);
a21_40 = sim(netf2_40,p2);

netf2_100 = feedforwardnet(10,'trainlm');
netf2_100.trainParam.epochs=100;
netf2_100= train(netf2_100,p2,t2);
a21_100 = sim(netf2_100,p2);

% Plotting
%%%%%%%%%%

figure;
figure('Color',[1 1 1]);
% function f
subplot(2,3,1);
plot(xx1,ff,'b',x1,cell2mat(a11_4),'--g', x1, cell2mat(a11_10), '--r');
h_legend = legend('true target', 'trainlm 4', 'trainlm 10');
set(h_legend,'FontSize',14);
subplot(2,3,2);
postregm(cell2mat(a11_4),f);
subplot(2,3,3);
postregm(cell2mat(a11_10),f);

% function g
subplot(2,3,4);
plot(xx2,gg,'b',x2,cell2mat(a21_40),'--g', x2, cell2mat(a21_100), '--r'); 
h_legend = legend('true target', 'trainlm 40', 'trainlm 100'); 
set(h_legend,'FontSize',14);
subplot(2,3,5);
postregm(cell2mat(a21_40), g);
subplot(2,3,6);
postregm(cell2mat(a21_100), g);

export_fig('trainlm.pdf')