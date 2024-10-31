# Jax_Neural_network

This is the implementation of solving PDE in [Jax](https://github.com/jax-ml/jax).

## Equation
$$
\begin{aligned}
    \partial_t u + u \, \partial_x u - (0.01/\pi) \, \partial_{xx} u &= 0, \quad &&\quad (t,x) \in (0,1] \times (-1,1),\\
   u(0,x) &= - \sin(\pi \, x),                 \quad &&\quad x \in [-1,1],\\
   u(t,-1) = u(t,1) &= 0,                      \quad &&\quad t \in (0,1].
\end{aligned}
$$


## Result

the loss is 8.559e-04.


![image](https://github.com/user-attachments/assets/349c47c8-d1f4-499e-a884-8e662099959d)

for t = 0.9:

![image](https://github.com/user-attachments/assets/5d313c60-edef-4fa2-98d3-64674e400266)



