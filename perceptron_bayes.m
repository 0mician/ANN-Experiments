% Generate a single neuron perceptron with zero bias and two arbitrary weights
% plot the targets and decision boundary
numinput=10;
net=newp([-1 1; -1 1], 1);
net.IW{1,1}=rands(1,2);
P=rands(2,numinput);
T=sim(net,P);
subplot(1,2,1);
plotpv(P,T);
hold on;
plotpc(net.IW{1,1},0);

% Generate a prior distribution for the weights and plot it
w1=(-1:0.1:1)';
w2=(-1:0.1:1)';
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2;
    end
end
subplot(1,2,2);
surf(w1,w2,prior);

% Create posteriors by presenting all targets one by one.
% Plot updated distribution after each update.
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
    surf(w1,w2,prior);
    pause(1);
end
