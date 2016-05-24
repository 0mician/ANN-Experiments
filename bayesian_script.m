clear, clc, close all;

x = -pi:0.1:pi; y = sin(x);
fn = y + randn(size(y))*0.2;
p= con2seq(y); t = con2seq(fn); 

figure('Color', [1 1 1]);
subplot(2,2,1);
plot(x, y, 'k', x, fn, 'bx');
title('Dataset','FontSize',18,'FontWeight', 'normal');
xlabel('x','FontSize',16); ylabel('sin(x)','FontSize',16);
legend('sin(x)', 'noise');

neurons=[2 4 8];
for i=2:4
    % %creation of networks
    net1=feedforwardnet(neurons(i-1),'trainlm');
    net1.trainParam.showWindow = false;
    net2=feedforwardnet(neurons(i-1),'trainbr');
    net2.trainParam.showWindow = false;
    
    net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks
    net2.lw{2,1}=net1.lw{2,1};
    net2.b{1}=net1.b{1};
    net2.b{2}=net1.b{2};
    
    net1.trainParam.epochs=100;
    net2.trainParam.epochs=100;
    net1=train(net1,p,t);
    net2=train(net2,p,t);
    a13=sim(net1,p); a23=sim(net2,p);
    
    %plots
    subplot(2,2,i);
    plot(x,y,'k',x,cell2mat(a13),'rx',x,cell2mat(a23),'go');
    title(['Number of neurons: ' num2str(neurons(i-1)) '  (100 epochs)'],'FontSize',18,'FontWeight', 'normal');
    xlabel('x','FontSize',16); ylabel('sin(x)','FontSize',16);
    legend('sin(x)','trainlm','trainbr',4);
end

export_fig('bayes_trainbrlm.pdf');
