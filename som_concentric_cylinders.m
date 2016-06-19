% Marco Signoretto, March 2011
close all; clear all; clc;

% first we generate data uniformely distributed within two
% concentric cylinders
X=2*(rand(5000,3)-.5);
indx=(X(:,1).^2+X(:,2).^2<.6)&(X(:,1).^2+X(:,2).^2>.1);
X=X(indx,:)';

% we then initialize the SOM with hextop as topology function
% and linkdist as distance function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nethex  = newsom(X,[5 5 5],'hextop','linkdist'); 
netgrid = newsom(X,[5 5 5],'gridtop','linkdist'); 
netrand = newsom(X,[5 5 5],'randtop','linkdist'); 

% plot the data distribution with the prototypes of the untrained network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Color', [1 1 1]);
subplot(3,4,1);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-2 2 -2 2]);
hold on
plotsom(nethex.iw{1,1},nethex.layers{1}.distances)
title('Hexagonal topology initial','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off

subplot(3,4,5);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-2 2 -2 2]);
hold on
plotsom(netgrid.iw{1,1},netgrid.layers{1}.distances)
title('Grid topology initial','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off

subplot(3,4,9);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-2 2 -2 2]);
hold on
plotsom(netrand.iw{1,1},netrand.layers{1}.distances)
title('Random topology initial','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off

% finally we train the network and see how their position changes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hexagonal topology
nethex.trainParam.epochs = 1000;
nethex = train(nethex,X);
subplot(3,4,2);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(nethex.iw{1,1},nethex.layers{1}.distances)
title('Trained (X,Y)','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off
subplot(3,4,3);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(nethex.iw{1,1},nethex.layers{1}.distances)
title('Trained (X,Z)','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off

% Grid topology
netgrid.trainParam.epochs = 1000;
netgrid = train(netgrid,X);
subplot(3,4,6);
title('Grid topology','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(netgrid.iw{1,1},netgrid.layers{1}.distances)
title('Trained (X,Y)','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off
subplot(3,4,7);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(netgrid.iw{1,1},netgrid.layers{1}.distances)
title('Trained (X,Z)','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off

% Random topology
netrand.trainParam.epochs = 1000;
netrand = train(netrand,X);
subplot(3,4,10);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(netrand.iw{1,1},netrand.layers{1}.distances)
title('Trained (X,Y)','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off
subplot(3,4,11);
plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(netrand.iw{1,1},netrand.layers{1}.distances)
title('Trained (X,Z)','FontSize',18,'FontWeight', 'normal');
xlabel('W(i,1)','FontSize',14);
ylabel('W(i,2)','FontSize',14);
zlabel('W(i,3)', 'Fontsize', 14);
hold off
 