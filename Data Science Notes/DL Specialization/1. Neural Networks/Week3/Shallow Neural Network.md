
## Neural Networks Overview

## Neural Network Representation
Lets assume we only have one training sample having 3 features $x_1,x_2,x_3$ .


![[Pasted image 20250521091945.png]]

$z_1^{[1]} = w_1^{[1]T}x + b_1^{[1]}$ , $a_1^{[1]} = \sigma(z_1^{[1]})$  
$z_2^{[1]} = w_2^{[1]T}x + b_2^{[1]}$ , $a_2^{[1]} = \sigma(z_2^{[1]})$  
$z_3^{[1]} = w_3^{[1]T}x + b_3^{[1]}$ , $a_3^{[1]} = \sigma(z_3^{[1]})$  
$z_4^{[1]} = w_4^{[1]T}x + b_4^{[1]}$ , $a_4^{[1]} = \sigma(z_4^{[1]})$  

After vectorization this becomes:

$$
\begin{bmatrix}
z_1^{[1]} \\
z_2^{[1]} \\
z_3^{[1]} \\
z_4^{[1]}
\end{bmatrix}_{4 \times 1}
=
\begin{bmatrix}
-- w_1^{[1]T} -- \\
-- w_2^{[1]T} -- \\
-- w_3^{[1]T} -- \\
-- w_4^{[1]T} --
\end{bmatrix}_{4 \times 3}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}_{3 \times 1} +
\begin{bmatrix}
b_1^{[1]} \\
b_2^{[1]} \\
b_3^{[1]} \\
b_4^{[1]}
\end{bmatrix}_{4 \times 1}
$$
And,

$$
\begin{bmatrix}
a_1^{[1]} \\
a_2^{[1]} \\
a_3^{[1]} \\
a_4^{[1]}
\end{bmatrix}_{4 \times 1} =
\sigma
\begin{bmatrix}
z_1^{[1]} \\
z_2^{[1]} \\
z_3^{[1]} \\
z_4^{[1]}
\end{bmatrix}_{4 \times 1}
$$

In compressed notation these two equations become:

$z^{[1]} = W^{[1]}x + b^{[1]}$
$a^{[1]} = \sigma(z^{[1]})$

Similarly for 2$^{nd}$ layer:

$z^{[2]} = W^{[1]}a^{[1]} + b^{[2]}$
$a^{[2]} = \sigma(z^{[2]})$

## Vectorizing Across Multiple Examples
Now to do the above thing for $m$ training samples and each training sample having $n_x$ features.
for i=1 to m,
	$z^{[1](i)} = W^{[1]}x^{(i)}+b^{[1]}$
	$a^{[1](i)} = \sigma(z^{[1](i)})$
	$z^{[2](i)} = W^{[2]}a^{[1](i)}+b^{[2]}$
	$a^{[2](i)} = \sigma(z^{[2](i)})$

To vectorize this we do:

$$
\begin{bmatrix}
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
z^{[1](1)} & z^{[1](2)} & \dots & z^{[1](m)} \\
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
\end{bmatrix}_{4 \times m} = W^{[1]}
\begin{bmatrix}
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
x^{(1)} & x^{(2)} & \dots & x^{(m)} \\
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
\end{bmatrix}_{n_x \times m} + b^{[1]}
$$

Also,
$$A^{[1]}=
\begin{bmatrix}
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
a^{[1](1)} & a^{[1](2)} & \dots & a^{[1](m)} \\
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
\end{bmatrix}_{4 \times m}
$$

Compressed notation for these equations:
$Z^{[1]} = W^{[1]}X + b^{[1]}$
$A^{[1]} = \sigma(Z^{[1]})$
$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$
$A^{[2]} = \sigma(Z^{[2]})$

NOTE: attach vertically for multiple nodes combining of same layer and attach horizontally for multiple samples combining

## Activation Functions
Till now we have been using sigmoid function as our activation function. but we can have different function as well
but it should be non linear
### Sigmoid

$$
a = \frac{1}{1+e^{-z}}
$$
$$
\frac{da}{dz}=a(1-a)
$$


![[Pasted image 20250521101602.png]]

* Only used in binary classification output layer
### Tanh

![[Pasted image 20250521101631.jpg]]


$$
=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
$$
$$
\frac{da}{dz} = 1-a^2
$$

* Other than the above case tanh() is always better than sigmoid
* Mathematically its a shifted version of sigmoid
### ReLu

* Rectified Linear Unit
* Model learns faster
* Most commonly used

![[Pasted image 20250521102226.png]]

$$
a=max(0,z)
$$
$$
\frac{da}{dz}=
\begin{cases}
0, & \text{if } z < 0 \\
1, & \text{if } z \geq 0
\end{cases}
$$
* derivative is undefined for z $\geq$ 0, but we use 1 for computation

### Leaky ReLu


![[Pasted image 20250521102647.jpg]]

$$
a=max(0.01,z)
$$

## Why do you need Non-Linear Activation Functions?

* Also called identity activation function
* watch video again

## Gradient descent for neural networks

