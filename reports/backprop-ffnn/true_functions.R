library(ggplot2)
library(gridExtra)

# function 1
x <- seq(-5.0,5.0,0.01)
y <- x^3 - 11*x +2
p1 <- ggplot() +
  geom_line(aes(x,y)) +
  xlab("x") +
  ylab("f(x)") +
  ggtitle("Polynomial of order 3")

# function 2 
xx <- seq(-3.14,3.14,0.01)
yy <- exp(-xx^2) * sin(5*xx)
p2 <- ggplot() +
  geom_line(aes(xx,yy)) +
  xlab("x") +
  ylab("g(x)") +
  ggtitle("Dampen wiggling function")

# plot side-by-side
grid.arrange(p1, p2, ncol=2)
ggsave("figures/true_functions.pdf", arrangeGrob(p1, p2, ncol = 2))

