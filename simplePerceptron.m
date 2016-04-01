function simplePerceptron()
% small library of function to interact with perceptrons

    function [net] = setAndTrainPerceptron(input, target)
    % returns a trained perceptron (if linearly separable target)
        net = perceptron();
        net = configure(net, input, target);
        net.IW{1,1} = rand([1 2]);
        net.b{1} = rand();
        net = train(net, input, target);
    end

    function [net, input, target] = dummyPerceptron()
    % returns a simple perceptron (dataset linearly separable
        input = [2 1 -2 -1; 2 -2 2 1];
        target = [0 1 0 1];
        net = setAndTrainPerceptron(input, target);
    end

    function [net, input, target] = unclassifiableWithPerceptron()
    % example where perceptron cannot classify (XOR problem)
        input = [-1 -1 1 1; -1 1 -1 1];
        target = [1 0 0 1];
        net = setAndTrainPerceptron(input, target);
    end

    function displayPerceptron(input, target, net)
        hold on;
        plotpv(input, target);
        plotpc(net.IW{1,1}, net.b{1});
    end

    function [t] = perceptronTeacherStudent()
    % generates dataset from 1 perceptron, and feeds it to another for
    % training
       [net] = dummyPerceptron();
       n = 150;
       p = 2.*rands(2, n);
       t = sim(net, p);
       displayPerceptron(p, t, net);
       
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

% small user script to interact with the functions defined in simplePerceptron
choice = -1;

while choice ~= 0
    disp('0. Exit simulation');
    disp('1. See a working classification');
    disp('2. See a failing classification (XOR - 1000 iterations)');
    disp('3. Create a dataset from a perceptron and use it for training of another perceptron');
    choice = input('What would you like to do? ');
    
    close all;
    nntraintool close;
    
    switch choice
        case 1
            [net, P, T] = dummyPerceptron();
            displayPerceptron(P, T, net);
        case 2
            [net, P, T] = unclassifiableWithPerceptron();
            displayPerceptron(P, T, net);
        case 3
            t = perceptronTeacherStudent();
        otherwise
            disp('Bye')
    end
end
end