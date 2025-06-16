
## Logistic regression

L$_2$ Regularization:
$$
J(w,b) = \frac{1}{m} \sum_{i=1}^m L\big(\hat{y}^{(i)}, y^{(i)}\big)+
\frac{\lambda}{2m} {\vert\vert W \vert\vert}^2_2
$$
Here,
${\vert\vert W \vert\vert}$ is called "norm" of W. In L2 regularization we use norm w square
$$
{\vert\vert W \vert\vert}^2_2 = \sum_{j=1}^{n_x}W_j^2 = W^TW
$$

Note: Why dont we regularize the parameter b ($+ \frac{\lambda}{2m}b^2$)?
Ans: because is just a single number, it doesnt make much difference, but we can do it if we want

L$_1$ Regularization:

$$
J(w,b) = \frac{1}{m} \sum_{i=1}^m L\big(\hat{y}^{(i)}, y^{(i)}\big)+
\frac{\lambda}{2m} {\vert\vert W \vert\vert}_1
$$
W will end up being sparse, means there will be a lot of 0s in W.

$\lambda$ is called regualrization parameter
	try values to tune


## Neural network

$$
J(w^{[l]}, b^{[l]}) = \frac{1}{m} \sum_{i=1}^m L\big(\hat{y}^{(i)}, y^{(i)}\big)+
\frac{\lambda}{2m} \sum_{l=1}^L {\vert\vert W^{[L]} \vert\vert}_F^2
$$

Frobenius Norm of matrix (Hence the subscript F)
$$
{\vert\vert W^{[L]} \vert\vert}_F^2 = \sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{[l-1]}}(W^{[l]}_{ij})^2
$$

## How does regularization prevent overfitting ?


 



## Dropout regularization

![[Pasted image 20250521214310.png]]

* Toss a coin and 0.5 chance we remove the node and 0.5 chance we keep that node
* Remove links to the removed nodes
* Back propagate and train
* Using original network do toss again and remove different set of nodes
* Do new tosses for each training sample and train the model
### Implementing dropout ("Inverted dropout")

illustrate this with layer l=3
set a vector `d3 = np.random.rand(a3.shape[0],a3.shape[1]) < keep_prob
`a3 = np.multiply(a3.d3)`
`a3/ = keep_prob`
watch video again


### Making predictions at test time

* We don't do dropout at time of test

## Other Regularization Methods

### Data augmentation

Make fake training examples by:
	flipping the images
	zooming the images
	add distortions
	
they dont add as much info as a brand new examples, but it is cheap.
this also regularizes data
### Early Stopping

stop training before it overfits