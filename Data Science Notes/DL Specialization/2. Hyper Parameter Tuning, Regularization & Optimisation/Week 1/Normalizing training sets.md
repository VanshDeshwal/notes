

## Why normalize inputs


![[Pasted image 20250522150617.png]]

* normalized cost function is easier to optimize
* algo will learn faster
* no harm in normalizing

## Vanishing/ Exploding Gradients

![[Pasted image 20250524093845.png]]

Lets say we have a deep neural network like this, for simplicity let the activation function be linear, also let b for each layer =0.

then $\hat y = w^{[1]} \times w^{[2]} \times \dots \times w^{[L]} \times X$ = $w^{L} \times X$ 
so if w= 1.5 then $\hat y$ explodes(becomes very large) and if w < 1 then $\hat y$ vanishes .
When finding gradients we take derivatives of ($y-\hat y$) so a similar argument can be made for gradients.



## Weight Initialization for Deep Networks

A partial solution to the above problem is weight initialization.
If we use ReLu activation function then use:
$$
w^{[l]} = np.random.randn(shape)*\boxed {np.sqrt(\frac{2}{n^{[l-1]}})}
$$
If we are using tanh activation function then use this box
Xavier initialization: 
$$
\boxed{\sqrt {\frac{2}{n^{[l-1]}+n^{[l]}}}}
$$
## Numerical Approximation of Gradients

![[Pasted image 20250522153040.png]]

this plot is $f(\theta) = \theta^3$

usually we use only one triangle, either $\theta + \epsilon$ or $\theta - \epsilon$ base. in those cases we get:
$$
\frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}
$$
$$
\frac{(1.01)^3-(1)^3}{0.01} = 3.0301
$$
error = 0.03
$\epsilon$ = 0.01
So, in this case we get an error of order O($\epsilon$)
But if we take the both sided difference we get:

$$
\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2 \epsilon} \approx
g(\theta)
$$

$$
\frac{(1.01)^3 - (0.99)^3} {2 (0.01)}=
3.0001 \approx 3
$$
error = 0.0001
$\epsilon$ = 0.01, $\epsilon^2$ = 0.0001
we get an error of order O($\epsilon^2)$ 
## Gradient Checking

We use this technique to check if our implementation of back propagation is correct or not.

S1: reshape all your parameters into  vectors then concatenate those vectors

Take $W^{[1]},b^{[1]},\dots,W^{[L]},b^{[L]}$ and reshape into a big vector $\theta$.

So, our cost function becomes a function of $\theta$ = $J(\theta)$ 

Take $dW^{[1]},db^{[1]},\dots,dW^{[L]},db^{[L]}$ and reshape into a big vector $d\theta$.

Is $d\theta$ the slope of $J(\theta)$ ?

`for each i :`
$$
d\theta_{approx}[i] = \frac{J(\theta_1,\theta_2,\dots,\theta_i+\epsilon,\dots)-J(\theta_1,\theta_2,\dots,\theta_i-\epsilon,\dots)}{2\epsilon}
$$
$$
\approx d\theta[i] = \frac{\partial J}{\partial \theta_i}
$$
Check 
$$
\frac{\vert\vert d\theta_{approx} - d\theta \vert\vert_2}{\vert\vert d\theta_{approx}\vert\vert_2 + \vert\vert d\theta\vert\vert_2} \approx 10^{-7}
$$

if $\approx 10^{-7}$ then great!!
if $\approx 10^{-5}$ then something might be wrong
if $\approx 10^{-3}$ then some big problem

Rules:
* Don't use in training - only to debug
* If algorithm fails grad check, look at components to try identify bug
* Remember regularization
* Doesn't work with dropout
* Run at random initialization; perhaps again after some training

