
## Mini-batch gradient descent

We know that vectorization allows us to process all m examples efficiently without using for loop
In gradient descent we take one step after processing all m examples, if m is very large it takes a lot of time.
So, we divide our training set into mini batches
$$
X_{(n_x,m)} = [\underbrace{x^{(1)}x^{(2)}x^{(3)}\dots x^{(1000)}}_{X^{\{1\}}}\vert \underbrace{ x^{(1001)}\dots x^{(2000)}}_{X^{\{2\}}}\vert\dots\dots\vert\dots x^{(m)}]
$$
$$
Y_{(1,m)} = [\underbrace{y^{(1)}y^{(2)}y^{(3)}\dots y^{(1000)}}_{Y^{\{1\}}}\vert \underbrace{ y^{(1001)}\dots y^{(2000)}}_{Y^{\{2\}}}\vert\dots\dots\vert\dots y^{(m)}]
$$
Notations:
$X^{(i)}$ : $i^{th}$ training example
$Z^{[l]}$ : $l^{th}$ layer of neural network
$X^{\{t\}},Y^{\{t\}}$: t$^{th}$ mini batch

### How it works?

Lets take m = 5,000,000
then if batch of each size is 1000
we will have 5000 batches

i think L is not for layers in this algo, watch video again for clarity

Repeat {
	for t = 1,......,5000 {
		Forward prop on $X^{\{t\}}$:
			$Z^{[1]} = W^{[1]}X^{\{t\}}+b^{[1]}$
			$A^{[1]} = g^{[1]}(Z^{[1]})$
			$\vdots$
			$A^{[L]} = g^{[L]}(Z^{[L]})$
		Compute cost function: 
			$J^{\{t\}}=\frac{1}{1000}\sum_{i=1}^{l}L(\hat y^{(i)},y^{(i)})+\frac{\lambda}{2(1000)}\sum_{l}\vert\vert W^{[l]}\vert\vert_F^2$
		Backprop to compute gradients wrt $J^{\{t\}}$
		Update the weights:
			$W^{[l]}:=W^{[l]}-\alpha dW^{[l]}$
			$b^{[l]}:=b^{[l]}-\alpha db^{[l]}$
	}
}		

## Training with mini  batch gradient descent

![[Pasted image 20250526110447.png]]

In case of batch gradient descent the cost function J always decreases after every iteration\
But, for mini batch gradient descent J trends downwards but might increase in some iterations

## Choosing your mini-batch size

If mini-batch size = m : Batch gradient descent
If mini-batch size = 1  : Stochastic gradient descent

![[Pasted image 20250526110751.png]]

Batch gradient descent : too long per iteration
Stochastic gradient descent : loose speed that gained from vectorization
In-between :  fastest learning

### Guidelines for choosing mini-batch size

* It small training set (m<=2000): use batch gradient descent
* Typical mini-batch size : 64, 128, 256, 512, 1024
* Make sure mini-batch size fits in CPU/GPU memory

## Exponentially weighted averages

Below are some temperature recordings on a given day in london.

$\theta_1=40^\circ F$ 
$\theta_2=49^\circ F$ 
$\theta_3=45^\circ F$ 
$\vdots$
$\theta_{180}=60^\circ F$ 
$\theta_{181}=56^\circ F$ 
$\vdots$

If we plot these we get this graph:
![[Pasted image 20250526112631.png]]

$V_0 = 0$
$V_1=0.9(V_0) + 0.1(\theta_1)$
$V_2=0.9(V_1) + 0.1(\theta_2)$
$V_3=0.9(V_2) + 0.1(\theta_3)$
$\vdots$
$V_t=0.9(V_{t-1}) + 0.1(\theta_t)$


If we plot these weighted averages we get his red line:
![[Pasted image 20250526112930.png]]

$V_t=(\beta) V_{t-1} + (1-\beta)\theta_t$

watch video after this

$V_{100}=0.9(V_{99}) + 0.1(\theta_{100})$
$V_{99}=0.9(V_{98}) + 0.1(\theta_{99})$
$V_{98}=0.9(V_{97}) + 0.1(\theta_{98})$
$\vdots$

$V_{100}=0.1(\theta_{100}) + 0.9\Big[0.1(\theta_{99}) + 0.9\Big[0.1(\theta_{98})+ 0.9\Big[\dots$

## Implementing exponentially weighted averages

$V_\theta=0$
repeat{
	Get next $\theta_t$
	$V_\theta=\beta V_\theta + (1-\beta)\theta_t$
}

Advantages: takes very less memory, one line of code
Disadvantage: not great way of taking average, worse then normal average
but normal average is expensive

## Bias Correction in Exponentially Weighted Averages

What is bias in this? watch video

How to correct bias?

$$
V_t = \frac{V_t}{1-\beta^t}
$$
for initial values of $t$ it will show a large difference but further it wont make any difference

In practice people don't use this much
## Gradient descent with momentum

This algo works faster than gradient descent

On iteration t:
	Compute $dW,db$ on current mini-batch
	$V_{dW} = (\beta)V_{dW} + (1-\beta)dW$
	$V_{db} = (\beta)V_{db} + (1-\beta)db$
	$W:=W-\alpha V_{dW}$
	$b:=b-\alpha V_{db}$

Hyperparameters: $\beta$ = 0.9
## RMSprop

On iteration t:
	Compute $dW,db$ on current mini-batch
$$
S_{dW} = \beta S_{dW} + (1-\beta)dW^2
$$
$$
S_{db} = \beta S_{db} + (1-\beta)db^2
$$
$$
W:=W-\alpha \frac{dW}{\sqrt{S_{dW}}}
$$
$$
b:=b-\alpha \frac{db}{\sqrt{S_{db}}}
$$



## Adam optimization algorithm

Adam : Adaptive Moment Estimation
Combination of Momentum and RMSprop

initialize:
$V_{dw}=0, S_{dw}=0, V_{db}=0, S_{db}=0$
On iteration t:
	Compute $dw,db$ using current mini batch
$$V_{dW} = \beta_1V_{dW} + (1-\beta_1)dW,$$
$$V_{db} = \beta_1V_{db} + (1-\beta_1)db$$$$S_{dW} = \beta_2V_{dW} + (1-\beta_2)dW^2,$$
$$S_{db} = \beta_2V_{db} + (1-\beta_2)db^2$$
$$V_{dW}^{corrected} = \frac{V_{dW}}{1-\beta_1^t},$$
$$V_{db}^{corrected} = \frac{V_{db}}{1-\beta_1^t}$$
$$S_{dW}^{corrected} = \frac{S_{dW}}{1-\beta_2^t},$$
$$S_{db}^{corrected} = \frac{S_{db}}{1-\beta_2^t}$$
$$W:=W-\alpha \frac{V_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}+\epsilon}}$$
$$b:=b-\alpha \frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}+\epsilon}}$$

$\epsilon$ is there to handle cases with denominator 0.

### Hyperparameters choice:
* $\alpha$ : needs to be tuned
* $\beta_1$ : 0.9
* $\beta_2$ : 0.999
* $\epsilon$ : 10$^{-8}$

## Learning Rate Decay


$$
\alpha = \frac{1}{1+decayRate\times epochNumber}\alpha_0
$$

## The problem of Local Optima
