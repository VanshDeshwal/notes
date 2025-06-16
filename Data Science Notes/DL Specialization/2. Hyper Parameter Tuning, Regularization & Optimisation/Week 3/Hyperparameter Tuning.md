Top priority:
	$\alpha$
Second priority:
	$\beta$
	#hidden units
	mini-batch size
Third priority:
	#layers
	learning rate decay
$\beta_1,\beta_2,\epsilon$

## Try random values: Don't use a grid

![[Pasted image 20250526212536.png]]

If hyper parameter 1 is alpha and hyper parameter 2 is epsilon then we know that alpha is very important but using grid we have tried 25 value pairs but only 5 values of alpha which has the most impact.

![[Pasted image 20250526212715.png]]

If we choose randomly we may still have 25 pairs but we are using 25 different values of alpha

## Coarse to fine

![[Pasted image 20250526212911.png]]

If we get to know that the circled points are working well then we can zoom in a smaller region and sample more densely in that blue square.

## Using an appropriate scale to pick Hyperparameters

sampling at random doesn't mean sampling uniformly at random, 
over the range of valid values. 
Instead, it's important to pick the appropriate scale 
on which to explore the hyperparameters. 

Let's say that you're trying to choose the number of hidden units, $n[l]$, for a given layer $l$. 

And let's say that you think a good range of values is somewhere from 50 to 100. 

In that case, if you look at the number line from 50 to 100, 

maybe picking some number of  values at random within this number line. 

There's a pretty visible way to search for this particular hyperparameter. 
Or if you're trying to decide on the number of layers in your neural network, 
we're calling that $L$. 
Maybe you think the total number of layers should be somewhere between 2 to 4. 
Then sampling uniformly at random, along 2, 3 and 4, might be reasonable. 
Or even using a grid search, where you explicitly evaluate the values 2, 3 and 4 might be reasonable. 
So these were a couple examples where sampling uniformly at random over the range you're contemplating; might be a reasonable thing to do. 
But this is not true for all hyperparameters. 

Let's look at another example.

