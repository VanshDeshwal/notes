## Binary Classification

The goal is to train a classifier for which the input is an image represented by a feature vector, 𝑥, and predicts whether the corresponding label𝑦is 1 or 0. In this case, whether this is a cat image(1)or a non-cat image

![[Pasted image 20250519175017.png]]

An image is stored in the computer in three separate matrices corresponding to the Red, Green, and Blue color channels of the image. The three matrices have the same size as the image, for example, the resolution of the cat image is 64 pixels X 64 pixels, the three matrices (RGB) are 64 X 64 each. The value in a cell represents the pixel intensity which will be used to create a feature vector of $n$ dimension. In pattern recognition and machine learning, a feature vector represents an image, Then the classifier's job is to determine whether it contain a picture of a cat or not. To create a feature vector, 𝑥, the pixel intensity values will be “unrolled” or “reshaped” for each color. The dimension of the input feature vector𝑥 is𝑛= 64𝑥 64𝑥3 = 12288



`64*64*3= 12288`

![[Pasted image 20250519175136.png]]

### Notations
$(x,y)$ is a single training example, $x \in  \mathbb{R}^{n_x}$, $y \in$ {0,1}

we have 'm' training examples : { ($x^{(1)}$,$y^{(1)}$) , ($x^{(2)}$,$y^{(2)}$) , ...... , ($x^{(m)}$,$y^{(m)}$) }


$$X =
\begin{bmatrix}
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
x^{(1)} & x^{(2)} & \dots & x^{(m)} \\
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
\end{bmatrix}
$$

this is a matrix of m columns and  $n_x$ rows

$$
Y = 
\begin{bmatrix}
y^{(1)} & y^{(2)} & \dots & y^{(m})
\end{bmatrix}
$$$Y \in \mathbb{R}^{1 \times m}$
`X.shape =`$( n_x,m )$ 
`Y.shape =`$(1,m)$


## Logistic Regression

Logistic regression is a learning algorithm used in a supervised learning problem when the output 𝑦 are all either zero or one. The goal of logistic regression is to minimize the error between its predictions and training data. 

Example: Cat vs No - cat 
Given an image represented by a feature vector 𝑥, the algorithm will evaluate the probability of a cat being in that image. 

𝐺𝑖𝑣𝑒𝑛 𝑥 , 𝑦̂ = 𝑃(𝑦 = 1|𝑥), where 0 ≤ 𝑦̂ ≤ 1

The parameters used in Logistic regression are:
* The input features vector: 𝑥 ∈ $\mathbb{R}^{n_x}$, where $n_x$ is the number of features
* The training label: 𝑦 ∈ 0,1
* The weights: 𝑤 ∈ $\mathbb{R}^{n_x}$, where $n_x$ is the number of features
* The threshold: 𝑏 ∈ $\mathbb{R}$
* The output: 𝑦̂ = 𝜎($𝑤^𝑇𝑥 + 𝑏$)
* Sigmoid function: s = 𝜎($𝑤^𝑇𝑥 + 𝑏$) = 𝜎(𝑧)= $\frac{1}{1 + e^{-x}}$

## Logistic Regression cost function

$\hat{y}^{(i)} = \sigma(w^Tx^{(i)} + b)$ , where $\sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}}$

Given  { ($x^{(1)}$,$y^{(1)}$) , ($x^{(2)}$,$y^{(2)}$) , ...... , ($x^{(m)}$,$y^{(m)}$) }, we want $\hat{y}^{(i)} \approx y^{(i)}$

### Loss (error) function: 

The loss function measures the discrepancy between the prediction ( 𝑦̂$^{(𝑖)}$ ) and the desired output (𝑦$^{(𝑖)}$ ). In other words, the loss function computes the error for a single training example. 
$$
L\big(\hat{y}^{(i)}, y^{(i)}\big) = \frac{1}{2} \big(\hat{y}^{(i)} - y^{(i)}\big)^2
$$
We don't use this loss function because this will have multiple local minima, so instead we use:
$$
\boxed{
L\big(\hat{y}^{(i)}, y^{(i)}\big) = -\left( y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right)
}
$$

* If 𝑦$^{(𝑖)}$ = 1:
	* $L\big(\hat{y}^{(i)}, y^{(i)}\big) = -\log \hat{y}^{(i)} \quad \text{where } \hat{y}^{(i)} \approx 1$ (check with pdf notes)
* If 𝑦 (𝑖) = 0:
	* $L\big(\hat{y}^{(i)}, y^{(i)} = 0\big) = -\log \big(1 - \hat{y}^{(i)}\big)$ where $log(1 − 𝑦̂ ^{(𝑖)} )$ and $𝑦̂^{(𝑖)}$ should be close to 0


### Cost function
The cost function is the average of the loss function of the entire training set. We are going to find the parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function.
$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m L\big(\hat{y}^{(i)}, y^{(i)}\big)
$$
$$
\boxed{
J(w, b) = - \frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \log( \hat{y}^{(i)}) + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \Big]
}
$$

## Gradient Descent
We want to find $(w,b)$ that minimize $J(w,b)$ 
we initialize w,b to some initial value, for logistic regression we can take initial value as zeroes

Repeat {

$$
	w := w - \alpha \frac{\partial J(w,b)}{\partial w}
	$$
$$
	b := b - \alpha \frac{\partial J(w,b)}{\partial b}
	$$
}

## Logistic Regression Gradient Descent
Let's assume we only have one training example having two parameters $x_1$ and $x_2$. then the cost function will be same as the loss function.
To find the loss function we use forward propagation.
$$ z=w_1x_1 + w_2x_2 + b$$
$$\hat y = a=\sigma(z)$$

Then, our loss function $L$ will be as follows:

$$L(a,y) =  -\left( y \log(a) + (1 - y) \log (1 - a) \right) \tag{1}$$

Our aim is to find $a$ and $y$ such that Loss is minimum. For that we use gradient descent.
In our case to apply gradient descent we need $\frac{dL(a,y)}{dw_1}$,$\frac{dL(a,y)}{dw_2}$,$\frac{dL(a,y)}{db}$.
To find these derivatives easily we use back propogation.


$$\frac{dL(a,y)} {da} =  - \frac{d}{da}\Big[ y \log(a) + (1 - y) \log (1 - a) \Big]$$
$$\boxed{\frac{dL(a,y)} {da}= - \frac{y}{a} + \frac{1-y}{1-a}} \tag{2}$$

Now next step back we get,
$$\frac{dL(a,y)}{dz} =  \frac{dL(a,y)}{da} \times \frac{da}{dz}$$

We know that
$$a = \frac{1}{1-e^{-z}}$$
$$\frac {da}{dz} = \frac {1}{(1-e^{-z})^2} \times e^{-z}$ = $a^2 \times e^{-z}$$ 
$$a = \frac{1}{1- e^{-z}}$$
$$1 + e^{-z} = \frac{1}{a}$$
$$e^{-z} = \frac{1-a}{a}$$
$$a^2 \times \frac{1-a}{a} = a(1-a)$$

$$=\Big[- \frac {y}{a} + \frac{1-y}{1-a} \Big] \Big[a(1-a) \Big]$$
$$\boxed{\frac{dL(a,y)}{dz} = a-y} \tag{3}$$
One more step back,

$$\frac{dL(a,y)}{dw_1} = \frac{dL(a,y)}{dz}\times \frac{dz}{dw_1}$$

We know that,
$$z = w_1x_1 + w_2x_2 + b$$
$$\frac{dz}{dw_1} = x_1$$
$$\boxed{\frac{dL(a,y)}{dw_1} = (a - y)x_1} \tag{4}$$
Also for $w_2$,
$$\frac{dL(a,y)}{dw_2} = \frac{dL(a,y)}{dz}\times \frac{dz}{dw_2}$$
$$\frac{dz}{dw_2} = x_2$$
$$\boxed{\frac{dL(a,y)}{dw_2} = (a - y)x_2} \tag {5}$$
And for $db$,
$$\frac{dL(a,y)}{db} = \frac{dL(a,y)}{dz}\times \frac{dz}{db}$$
$$\frac{dz}{db} = 1$$
$$\boxed{\frac{dL(a,y)}{db} = \frac{dL(a,y)}{dz}} \tag{6}$$
Now we can apply gradient descent by repeatedly updating:

$$	w_1 := w_1 - \alpha \frac{L(a,y)}{dw_1}$$
$$	w_2 := w_2 - \alpha \frac{L(a,y)}{dw_2}$$
$$	b := b - \alpha \frac{L(a,y)}{db}$$
until two consecutive similar values of L are obtained.
## Logistic regression on 'm' examples

Now lets take $m$ training samples. our cost function will be the average of all $m$ loss functions.
$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m L\big(a^{(i)}, y^{(i)}\big)
$$
$$a^{(i)}=\hat y^{(1)}=\sigma(z^{(i)})=\sigma(w^Tx^{(i)}+b) $$
$$\frac{dJ(w,b)}{dw_1}=\frac{1}{m}\sum_{i=1}^m \frac{dL(a^{(i)},y^{(i)})}{dw_1}$$
