clear; clc; close all;

load fisheriris
X = meas(:,1:end-1);
true_labels = meas(:,end); 
temp = ones(150,1);
for i=51:100
    temp(i)=2;
end
for i=101:150
    temp(i)=3;
end
true_labels = temp;

%Training the SOM
%%%%%%%%%%%%%%%%%
x_length = 3;
y_length = 1;
net = newsom(X',[y_length x_length],'gridtop');
net.trainParam.epochs = 100000;
net = train(net,X');
figure('Color', [1 1 1]);
scatter3(X(:,1),X(:,2),X(:,3), 20, true_labels)%,'.g','markersize',10);
hold on
plotsom(net.iw{1,1},net.layers{1}.distances)
title('Iris dataset SOM','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off

% Assigning examples to clusters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputs = sim(net,X');
[~,assignment]  =  max(outputs);

%Compare clusters with true labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[AR,RI,MI,HI]=unsupervised_randindex(assignment,true_labels');

