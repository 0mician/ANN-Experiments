close all, clc, clear all;

fig1 = figure('Color', [1 1 1]);

recurrent_hopdigit(0.5, 50);
recurrent_hopdigit(1, 50);
recurrent_hopdigit(5, 50);

fig2 = figure('Color', [1 1 1]);

recurrent_hopdigit(2.5, 1);
recurrent_hopdigit(2.5, 2);
recurrent_hopdigit(2.5, 3);
recurrent_hopdigit(2.5, 4);
recurrent_hopdigit(2.5, 5);
recurrent_hopdigit(2.5, 10);
