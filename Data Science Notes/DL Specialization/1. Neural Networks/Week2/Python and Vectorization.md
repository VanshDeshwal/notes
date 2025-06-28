
## Vectorizing LR

$$w =
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_{n_x}
\end{bmatrix}_{n_x \times 1}
$$

$$w^T=
\begin{bmatrix}
w_1 & w_2 & \dots & w_{n_x}
\end{bmatrix}_{1 \times n_x}
$$
Now to predict $\hat y$ we have to calculate $z$ of all individual training samples and get $\hat y = \sigma(z)$ for all training samples.
So, for training sample one we do

$z^{(1)} = 𝑤^𝑇𝑥^{(1)} + 𝑏$

where $z^{(1)}$ and $b$ are individual values and $w^T$ and $x^{(1)}$ are matrices

Expanded form of equation:
$$z^{(1)} =
\begin{bmatrix}
w_1 & w_2 & \dots & w_{n_x}
\end{bmatrix}_{1 \times n_x}
\begin{bmatrix}
x^{(1)}_1 \\
x^{(1)}_2 \\
\vdots \\
x^{(1)}_{n_x}
\end{bmatrix}_{n_x \times 1}
+ b
$$

Similarly, for second training sample,
$z^{(2)} = 𝑤^𝑇𝑥^{(2)} + 𝑏$

Expanded form:
$$z^{(2)} =
\begin{bmatrix}
w_1 & w_2 & \dots & w_{n_x}
\end{bmatrix}_{1 \times n_x}
\begin{bmatrix}
x^{(2)}_1 \\
x^{(2)}_2 \\
\vdots \\
x^{(2)}_{n_x}
\end{bmatrix}_{n_x \times 1}
+ b
$$

For $m^{th}$ training sample.
$z^{(m)} = 𝑤^𝑇𝑥^{(m)} + 𝑏$
$$z^{(m)} =
\begin{bmatrix}
w_1 & w_2 & \dots & w_{n_x}
\end{bmatrix}_{1 \times n_x}
\begin{bmatrix}
x^{(m)}_1 \\
x^{(m)}_2 \\
\vdots \\
x^{(m)}_{n_x}
\end{bmatrix}_{n_x \times 1}
+ b
$$
Now, combining these equations we get,

$$ 
\begin{bmatrix}
z^{(1)} & z^{(2)} & \dots & z^{(m)}
\end{bmatrix}
= w^TX +
\begin{bmatrix}
b & b & \dots & b
\end{bmatrix}_{1 \times m}
$$
where,

$$X =
\begin{bmatrix}
x^{(1)}_1 & x^{(2)}_1 & \dots & x^{(m)}_1 \\
x^{(1)}_2 & x^{(2)}_2 & \dots & x^{(m)}_2 \\
x^{(1)}_3 & x^{(2)}_3 & \dots & x^{(m)}_3 \\
\vdots    & \vdots    & \ddots     & \vdots \\
x^{(1)}_{n_x} & x^{(2)}_{n_x} & \dots & x^{(m)}_{n_x} \\
\end{bmatrix}_{n_x \times m}
$$


$$Z=w^T
\begin{bmatrix}
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
x^{(1)} & x^{(2)} & \dots & x^{(m)} \\
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
\end{bmatrix}_{n_x \times m}
+
\begin{bmatrix}
b & b & \dots & b
\end{bmatrix}
$$$$=
\begin{bmatrix}
w^Tx^{(1)} & w^Tx^{(2)} & \dots & w^Tx^{(m})
\end{bmatrix}
+
\begin{bmatrix}
b & b & \dots & b
\end{bmatrix}
$$
$$=
\begin{bmatrix}
w^Tx^{(1)}+b & w^Tx^{(2)}+b & \dots & w^Tx^{(m})+b
\end{bmatrix}
$$
$$Z=
\begin{bmatrix}
z^{(1)} & z^{(2)} & \dots & z^{(m})
\end{bmatrix}
$$

Command in numpy:
`Z = np.dot(w.T,X) + b`


## Vectorizing Logistic Regression's Gradient Output  

$dz^{(1)} = a^{(1)} - y^{(1)}$
$dz^{(2)} = a^{(2)} - y^{(2)}$
$\vdots$
$dz^{(m)} = a^{(m)} - y^{(m)}$
So,
$$dZ=
\begin{bmatrix}
dz^{(1)} & dz^{(2)} & \dots & dz^{(m})
\end{bmatrix}
$$
$$A=
\begin{bmatrix}
a^{(1)} & a^{(2)} & \dots & a^{(m})
\end{bmatrix}
$$
$$Y=
\begin{bmatrix}
y^{(1)} & y^{(2)} & \dots & y^{(m})
\end{bmatrix}
$$
$$ dZ = A - Y =
\begin{bmatrix}
a^{(1)}-y^{(1)} & a^{(2)}-y^{(2)} & \dots & a^{(m)}-y^{(m)}
\end{bmatrix}
$$

$$db = \frac {1}{m} \sum_{i=1}^m dz^{(i)}$$
in code:
	`1/m np.sum(dz)`

$dw = \frac{1}{m}XdZ^T$

$$=\frac{1}{m}
\begin{bmatrix}
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
x^{(1)} & x^{(2)} & \dots & x^{(m)} \\
\vert & \vert &        & \vert \\
\vert & \vert &        & \vert \\
\end{bmatrix}
\begin{bmatrix}
z^{(1)} \\
z^{(2)} \\
\vdots \\
z^{(m)}
\end{bmatrix}
$$
$$=\frac{1}{m}
\begin{bmatrix}
x^{(1)}dz^{(1)} + x^{(2)}dz^{(2)} + \dots + x^{(m)}dz^{(m)}
\end{bmatrix}
$$
