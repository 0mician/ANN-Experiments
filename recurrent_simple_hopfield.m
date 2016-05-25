% attractors states
T = [1 1; -1 -1; 1 -1]';

% plotting the attractors
scatter(T(1,:),T(2,:));%T(1,:),T(2,:),T(3,:),'r*')
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('X');
ylabel('Y');

% create network
net = newhop(T);
[Y,Pf,Af] = net([],[],T);
Y

% create random starting points and view response
for i=1:20
    a = {rands(2,1)};
    [y,Pf,Af] = net({20},{},a);
    
    % view the hopfield response
    record = [cell2mat(a) cell2mat(y)];
    record
    start = cell2mat(a);
    hold on
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:))
end