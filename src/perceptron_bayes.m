clear; clc; close all;
addpath export_fig

% Creation of a perceptron (no bias) to generate training dataset
numinput=50;
net=newp([-1 1; -1 1], 1);
net.IW{1,1}=rands(1,2);
P=rands(2,numinput);
T=sim(net,P);

figure('Color',[1 1 1]);

% Generate a prior distribution for the weights and plot it
w1=(-1:0.1:1)';
w2=(-1:0.1:1)';
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2;
    end
end
subplot(2,2,2);
surf(w1,w2,prior);
title('Prior','FontSize',18,'FontWeight', 'normal')
xlabel('w1');
ylabel('w2');

% Create posterior (iterate over points in dataset)
for k=1:numinput
    x=P(:,k);
    for i=1:length(w1)
        for j=1:length(w2)
            w=[w1(i) w2(j)];
            y=1/(1+exp(-w*x));
            likelihood=y^T(k)*(1-y)^(1-T(k));
            prior(i,j)=likelihood*prior(i,j);
        end
    end
    n=sum(sum(prior)); % This loop is for normalization of the distribution
    for i=1:length(w1)
        for j=1:length(w2)
            prior(i,j)=prior(i,j)/n;
        end
    end
end

subplot(2,2,3);
surf(w1,w2,prior);
title('Posterior','FontSize',18,'FontWeight', 'normal')
xlabel('w1');
ylabel('w2');
    
% MAP weights selection and plotting of learned classifier on top of
% perceptron
prob=0;
for i=1:length(w1)
    for j=1:length(w2)
        if (prior(i,j)>prob)
            prob=prior(i,j);
            maxind=[i,j];
        end
    end
end

subplot(2,2,1);
plotpv(P,T); % plot labelled data
hold on;
perceptron = plotpc(net.IW{1,1},0); % plot perceptron classifier
title('Classifiers','FontSize',18,'FontWeight', 'normal')
xlabel('X1');
ylabel('X2');
bayes_classifier = plotpc([w1(maxind(1)), w2(maxind(2))],0);
set(bayes_classifier, 'Color', 'g');
h_legend = legend([perceptron, bayes_classifier],'perceptron', 'bayes classifier');
set(h_legend,'FontSize',14);

% Countour plot + real weights from the perceptron (asterisks)
subplot(2,2,4);
contour(w1, w2, prior);

x=net.IW{1,1};

subplot(2,2,4);
hold on;
plot(x(2), x(1), 'rx');
title('Contour plots and perceptron weights','FontSize',18,'FontWeight', 'normal')
xlabel('w1');
ylabel('w2');

% export_fig('perceptron_bayes.pdf')