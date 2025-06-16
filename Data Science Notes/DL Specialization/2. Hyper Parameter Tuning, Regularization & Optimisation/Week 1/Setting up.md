

## Train/ Dev/ Test sets

We should divide our data into 3 parts:
* Training set : train algo on this data
* Dev set : use multiple models on this set to find best model
* Test set : use best model for unbiased evaluation

## Bias / Variance

![[Pasted image 20250523122420.png]]

Lets take the example of Cat Classification, humans can classify cats with 0% error, So then if we get:
Train set error: 1%
Dev set error: 11%
Overfitting on training set : High variance

Train set error: 15%
Dev set error: 16%
Underfitting on training set : High bias

Train set error: 15%
Dev set error: 30%
High variance & High Variance

Train set error: 0.5%
Dev set error: 1%
Low variance & Low Variance

## Basic recipe for machine learning

S1: High bias ? (training set performance)
	try bigger network
	train longer
	try different architecture
	do till you fit training data very well
S2: High Variance ?(dev set performance)
	use more data
	regularization
	try different architecture
	go to S1





