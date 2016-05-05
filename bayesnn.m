% Script that tries to linearly classify 4 points using bayesian approaches
% to estimate the weights (no bias, non linearly separable points, gaussian prior)

clear; clc; close all;
addpath export_fig

a=1;
b=1;
s=0.1;
w1=(-a:s:b)';
w2=(-a:s:b)';

X2=[-5 -5; 5 5];
X4=[-5 -5; 5 5; 0 1; -1 0];
T2=[0 ; 1];
T4=[0 ; 1; 0; 1];

% New view angles for graphics
az = 52.5;
el = 30;

%**********************************
% using just the first 2 data points

%figure;
figure('Color',[1 1 1]);
subplot(2,2,1);
scatter([-5 0],[-5 1],'o');
hold;
scatter([ 5 -1],[5 0], 'x');
title('Points to classify','FontSize', 16)
xlabel('X1');
ylabel('X2');

subplot(2,2,2);
% make prior
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2;
    end
end
surf(w1,w2,prior)
grid on
box on
title('Prior','FontSize', 16)

% make posteriors
n=size(X2,1);
posterior = prior;
for k=1:n
    x=X2(k,:);
    for i=1:length(w1)
        for j=1:length(w2)
            w=[w1(i) w2(j)];
            y=1/(1+exp(-w*x'));
            likelihood=y^T2(k)*(1-y)^(1-T2(k));
            posterior(i,j)=likelihood*posterior(i,j);
        end
    end
end
subplot(2,2,3);
surf(w1,w2,posterior)
grid on
box on
title('Posterior after 2 points','FontSize', 16)
view(az, el);
% camroll(90)

%************************
% using all 4 data points
% make prior again
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2;
    end
end

% make posteriors
n=size(X4,1);
posterior = prior;
for k=1:n
    x=X4(k,:);
    for i=1:length(w1)
        for j=1:length(w2)
            w=[w1(i) w2(j)];
            y=1/(1+exp(-w*x'));
            likelihood=y^T4(k)*(1-y)^(1-T4(k));
            posterior(i,j)=likelihood*posterior(i,j);
        end
    end
end
subplot(2,2,4);
surf(w1,w2,posterior)
grid on
box on
title('Posterior after all points','FontSize', 16)
view(az, el);

% find the weights of classifier (using MAP) - we know the classifier goes
% through the origin, so we can easily find another point to draw it
% w1*x1 + w2*x2 = 0 <-> if x1 = n, x2 = -w2/w1*n

[maxA,ind] = max(posterior(:));
[m,n] = ind2sub(size(posterior),ind);
w1_value = w1(m);
w2_value = w2(n);
ratio = -w2_value/w1_value;
x1 = [5 -5]; x2 = [5*ratio -5*ratio];
subplot(2,2,1);
classifier = line(x1, x2, 'Color','g');
h_legend = legend(classifier, 'classifier');
set(h_legend,'FontSize',14);

export_fig('bayesnn.pdf')