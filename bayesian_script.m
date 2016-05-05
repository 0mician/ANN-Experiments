x1 = -5:0.1:5; f = x1.^3 + x1.^2 - 1;
fnoisy = f + randn(size(f)) * 20;
p1 = con2seq(x1); t1 = con2seq(fnoisy);

