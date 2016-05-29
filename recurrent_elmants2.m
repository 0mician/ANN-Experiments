%In this script an elman network is trained and tested in order to
%model a so called hammerstein model. The system is described like this:

% x(t+1) = 0.6x(t-1) + sin(u(t))
%y(t) = x(t);

%Elman network should be able to understand the relation between output
%y(t) and input u(t). x(t) is a latent variable representing the internal
%state of the system/

clc;
clear;
close all;

n=1000; %total number of samples

u(1)=randn; %random number drawn from a standard gaussian distribution
x(1)=rand+sin(u(1));
y(1)=.6*x(1);

for i=2:n

    u(i)=randn;
    x(i)=.6*x(i-1)+sin(u(i));
    y(i)=x(i);

end

figure('Color', [1 1 1]);
subplot(2,1,1);
plot(y);
xlabel('time');
ylabel('y');
export_fig('hammerstein_system.pdf');

n_te = 200;
T_test=y(end-n_te:end); %test set
X_test=u(end-n_te:end);

% Selection of the number of training examples
i = 1;
test_mse = zeros(1,16);
for tn = [50:50:800]
    X=u(1:tn); %training set
    T=y(1:tn);
    net = newelm(X,T,20); %create network
    net.trainParam.showWindow = false;
    net.divideFcn = 'dividetrain';
    net = train(net,X,T); %train network
    T_test_sim = sim(net,X_test); %test network
    test_mse(i) = sum((T_test - T_test_sim).^2)/size(T_test,2);
    i = i + 1;    
end

figure('Color', [1 1 1]);
subplot(2,2,1);
plot(50:50:800, test_mse);
title('Training size vs Test MSE','FontSize',18,'FontWeight', 'normal');
ylabel('Test MSE','FontSize',14); 
xlabel('Training set size','FontSize',14);


n_tr = 300;
X=u(1:n_tr); %training set
T=y(1:n_tr);

T_test=y(end-n_te:end); %test set
X_test=u(end-n_te:end);

% Selection of nn and nepochs
i = 1; j = 1;
test_mse = zeros(20, 20);

for nn=[5:5:100]
    for nepochs=[100:100:2000]
        net = newelm(X,T,nn); %create network
        net.trainParam.epochs = nepochs;
        net.trainParam.showWindow = false;
        net.divideFcn = 'dividetrain';
        net = train(net,X,T); %train network
        T_test_sim = sim(net,X_test); %test network
        test_mse(i, j) = sum((T_test - T_test_sim).^2)/size(T_test,2);
        j = j + 1;
    end
    i = i + 1;
    j = 1;
end

saved_mse = test_mse;
subplot(2,2,2);
surf(1:5:100, 100:100:2000,test_mse);
grid on;
title('Test MSE surface','FontSize',18,'FontWeight', 'normal');
ylabel('Epochs','FontSize',14); 
xlabel('Neurons','FontSize',14);
zlabel('Test MSE','FontSize',14);

% Plot results and calculate correlation coefficient between target and
% output
net = newelm(X,T,20); %create network
net.trainParam.epochs = 1100;
net.trainParam.showWindow = false;
net.divideFcn = 'dividetrain';
net = train(net,X,T); %train network
T_test_sim = sim(net,X_test); %test network

subplot(2,1,2);
plot(1:size(X_test,2),T_test,'r',1:size(X_test,2),T_test_sim,'b');
title('Prediction','FontSize',18,'FontWeight', 'normal');
xlabel('time','FontSize',14);
ylabel('y','FontSize',14);
legend('target','prediction',-1);

export_fig('elman_prediction.pdf');