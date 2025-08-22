uses: 
* Speech recognition
* Music generation
* Sentiment classification
* DNA sequence analysis
* Machine Translation
* Video activity recognition
* Name entity recognition

## Notations

Lets say we want to do name entity recognition. then if the input is as below, we will get 1 in the output for the words that are names and 0 for the words that are not names
$$
x: \underset{x^{<1>}}{\text{Harry}} \,
   \underset{x^{<2>}}{\text{Potter}} \,
   \underset{x^{<3>}}{\text{and}} \,
   \underset{x^{<4>}}{\text{Hermione}} \,
   \underset{x^{<5>}}{\text{Granger}} \,
   \underset{x^{<6>}}{\text{invented}} \,
   \underset{x^{<7>}}{\text{a}} \,
   \underset{x^{<8>}}{\text{new}} \,
   \underset{x^{<9>}}{\text{spell.}}
$$
$$
y :                 1 1 0 1 1 0 0 0 0
$$


$x^{(i)<t>}$ represents the t$^{th}$ element of i$^{th}$ training example
$y^{(i)<t>}$ represents the t$^{th}$ element of i$^{th}$ output example
$T_x^{(i)}$ represents the total elements in i$th$ training example
$T_y^{(i)}$ represents the total elements in i$th$ output example

Representing words
![[Pasted image 20250609104513.png]]

## Recurrent Neural Networks

### Why not a standard network ?

![[Pasted image 20250609104653.png]]
 length of xt is not fixed
 "harry" appears in 2nd position in one example, network might learn that if harry appear in 2nd position then only its a name.
 
### Recurrent neural network
#### Forward propagation

![[Pasted image 20250609105759.png]]


### Simplified Notations

![[Pasted image 20250609110128.png]]

## Back propagation through time

![[Pasted image 20250609110920.png]]

## Different types of RNN

till now we have seen T$_x$ = T$_y$ but that's not always the cases

### Examples of RNN architectures

![[Pasted image 20250609111811.png]]


## Language Model and Sequence Generation 
### What is language modelling ?

![[Pasted image 20250609112332.png]]
job of a language model is that we input a sentence and the model will estimate the probability of sequence of those words

### Language modelling with an RNN

![[Pasted image 20250609113250.png]]

## Sampling Novel Sequences

### Sampling a sequence from a trained RNN

![[Pasted image 20250609113907.png]]

![[Pasted image 20250609113931.png]]

### Character-level language model

![[Pasted image 20250609114154.png]]

very computation heavy
doesn't work well with past dependencies, means prediction based on past characters is not very good

### Vanishing gradients with RNNs

- One of the problems with naive RNNs that they run into **vanishing gradient** problem.
    
- An RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to optimize.
    
- Let's take an example. Suppose we are working with language modeling problem and there are two sequences that model tries to learn:
    
    - "The **cat**, which already ate ..., **was** full"
    - "The **cats**, which already ate ..., **were** full"
    - Dots represent many words in between.
- What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN is not very good at capturing very long-term dependencies like this.
