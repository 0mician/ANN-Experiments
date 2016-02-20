function nn_perceptron()
    function [net] = set_and_train_perceptron(input, target)
        net = perceptron();
        net = configure(net, input, target);
        net.IW{1,1} = rand([1 2]);
        net.b{1} = rand();
        net = train(net, input, target);
    end

    function [net, input, target] = dummy_perceptron()
        input = [2 1 -2 -1; 2 -2 2 1];
        target = [0 1 0 1];
        net = set_and_train_perceptron(input, target);
    end

    function [net, input, target] = unclassifiable_with_perceptron()
        input = [-1 -1 1 1; -1 1 -1 1];
        target = [1 0 0 1];
        net = set_and_train_perceptron(input, target);
    end

    function display_perceptron(net, input, target)
        hold on;
        plotpv(input, target);
        plotpc(net.IW{1,1}, net.b{1});
    end

    function [t] = perceptron_teacher_student()
       [net] = dummy_perceptron();
       n = 150;
       p = 2.*rands(2, n);
       t = sim(net, p);
       display_perceptron(net, p, t);
       
       nets = perceptron();
       nets = configure(net, p, t);
       nets.IW{1,1} = rand([1 2]);
       nets.b{1} = rand();
       nets.adaptParam.passes = 1;
       [nets,Y,error]=adapt(nets,p,t);
       linehandle=plotpc(nets.IW{1,1},nets.b{1});
       
       while sum(abs(error ))~=0
           [nets,Y,error]=adapt(nets,p,t);
           linehandle=plotpc(nets.IW{1,1},nets.b{1},linehandle);
           pause(1);
       end
    end

disp('1. See a working classification');
disp('2. See a failing classification (XOR - 1000 iterations)');
disp('3. Create a dataset for training');

choice = input('What would you like to do? ');
switch choice
    case 1
        [net, P, T] = dummy_perceptron();
        display_perceptron(P, T, net);
    case 2
        [net, P, T] = unclassifiable_with_perceptron();
        display_perceptron(P, T, net);
    case 3
        t = perceptron_teacher_student();
    otherwise
        disp('Bye')
end
pause(5);
close all;
end
