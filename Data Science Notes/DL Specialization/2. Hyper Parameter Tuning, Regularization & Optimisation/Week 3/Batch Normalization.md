
## Normalizing Activations in a Network

In logistic regression we saw that normalizing the input features could speed up the training process.

$$
\mu = \frac{1}{m}\sum X^{(i)}
$$
$$
X = X-\mu
$$
$$
\sigma^2 = \frac{1}{m}\sum X^{(i)^2}
$$
$$
X = \frac{X}{\sigma}
$$
### How to normalize a neural network ?

#### Can we normalize the values $a^{[1]},a^{[2]},\dots etc$ ?

In practice we normalize $z^{[l]}$.

### Implementing batch norm

given some intermediate values in NN. $z^{(1)},\dots\dots ,z^{(m)}$

$$
\mu = \frac{1}{m}\sum z^{(i)}
$$
$$
\sigma^2=\frac{1}{m}\sum(z^{(i)}-\mu)^2
$$
$$
z^{(i)}_{norm} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\epsilon}}
$$
$$
\overset{\sim}z^{(i)} = \gamma z^{(i)}_{norm} + \beta 
$$

Watch video again, something about inverted equation,identity etc

## Fitting Batch Norm into a Neural Network
























