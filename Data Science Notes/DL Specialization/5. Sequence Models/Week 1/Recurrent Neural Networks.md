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
- One of the problems with naive RNNs that they run into **vanishing gradient** problem.

- An RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to optimize.

- Let's take an example. Suppose we are working with language modeling problem and there are two sequences that model tries to learn:

  - "The **cat**, which already ate ..., **was** full"
  - "The **cats**, which already ate ..., **were** full"
  - Dots represent many words in between.

- What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN is not very good at capturing very long-term dependencies like this.

- As we have discussed in Deep neural networks, deeper networks are getting into the vanishing gradient problem. That also happens with RNNs with a long sequence size.   
  ![](Images/16.png)   
  - For computing the word "was", we need to compute the gradient for everything behind. Multiplying fractions tends to vanish the gradient, while multiplication of large number tends to explode it.
  - Therefore some of your weights may not be updated properly.

- In the problem we descried it means that its hard for the network to memorize "was" word all over back to "cat". So in this case, the network won't identify the singular/plural words so that it gives it the right grammar form of verb was/were.

- The conclusion is that RNNs aren't good in **long-term dependencies**.

- > In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- _Vanishing gradients_ problem tends to be the bigger problem with RNNs than the _exploding gradients_ problem. We will discuss how to solve it in next sections.

- Exploding gradients can be easily seen when your weight values become `NaN`. So one of the ways solve exploding gradient is to apply **gradient clipping** means if your gradient is more than some threshold - re-scale some of your gradient vector so that is not too big. So there are cliped according to some maximum value.

  ![](Images/26.png)

- **Extra**:
  - Solutions for the Exploding gradient problem:
    - Truncated backpropagation.
      - Not to update all the weights in the way back.
      - Not optimal. You won't update all the weights.
    - Gradient clipping.
  - Solution for the Vanishing gradient problem:
    - Weight initialization.
      - Like He initialization.
    - Echo state networks.
    - Use LSTM/GRU networks.
      - Most popular.
      - We will discuss it next.
 
### Gated Recurrent Unit (GRU)
- GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long-term dependencies.

- The basic RNN unit can be visualized to be like this:   
  ![](Images/17.png)

- We will represent the GRU with a similar drawings.

- Each layer in **GRUs**  has a new variable `C` which is the memory cell. It can tell to whether memorize something or not.

- In GRUs,  $C^{<t>} = a^{<t>}$

- Equations of the GRUs:   
  ![](Images/18.png)
- The update gate is between 0 and 1
    - To understand GRUs imagine that the update gate is either 0 or 1 most of the time.
- So we update the memory cell based on the update cell and the previous cell.

- Lets take the cat sentence example and apply it to understand this equations:

  - Sentence: "The **cat**, which already ate ........................, **was** full"

  - We will suppose that U is 0 or 1 and is a bit that tells us if a singular word needs to be memorized.

  - Splitting the words and get values of C and U at each place:

| Word    | Update gate(U)             | Cell memory (C) |
| ------- | -------------------------- | --------------- |
| The     | 0                          | val             |
| cat     | 1                          | new_val         |
| which   | 0                          | new_val         |
| already | 0                          | new_val         |
| ...     | 0                          | new_val         |
| was     | 1 (I don't need it anymore)| newer_val       |
| full    | ..                         | ..              |

- Drawing for the GRUs   
  ![](Images/19.png)
  - Drawings like in http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is so popular and makes it easier to understand GRUs and LSTMs. But Andrew Ng finds it's better to look at the equations.
- Because the update gate U is usually a small number like 0.00001, GRUs doesn't suffer the vanishing gradient problem.
  - In the equation this makes $C^{<t>} = C^{<t-1>}$ in a lot of cases.
- Shapes:
  - $a^{<t>}$ shape is (NoOfHiddenNeurons, 1)
  - $c^{<t>}$ is the same as $a^{<t>}$
  - $c^{\sim<t>}$ is the same as $a^{<t>}$
  - $u^{<t>}$ is also the same dimensions of $a^{<t>}$
- The multiplication in the equations are element wise multiplication.
- What has been descried so far is the Simplified GRU unit. Let's now describe the full one:
  - The full GRU contains a new gate that is used with to calculate the candidate C. The gate tells you how relevant is $C^{<t-1>}$ to $C^{<t>}$
  - Equations:   
    ![](Images/20.png)
  - Shapes are the same
- So why we use these architectures, why don't we change them, how we know they will work, why not add another gate, why not use the simpler GRU instead of the full GRU; well researchers has experimented over years all the various types of these architectures with many many different versions and also addressing the vanishing gradient problem. They have found that full GRUs are one of the best RNN architectures  to be used for many different problems. You can make your design but put in mind that GRUs and LSTMs are standards.

### Long Short Term Memory (LSTM)
- LSTM - the other type of RNN that can enable you to account for long-term dependencies. It's more powerful and general than GRU.
- In LSTM , C<sup>\<t></sup> != a<sup>\<t></sup>
- Here are the equations of an LSTM unit:   
  ![](Images/21.png)
- In GRU we have an update gate `U`, a relevance gate `r`, and a candidate cell variables C<sup>\~\<t></sup> while in LSTM we have an update gate `U` (sometimes it's called input gate I), a forget gate `F`, an output gate `O`, and a candidate cell variables C<sup>\~\<t></sup>
- Drawings (inspired by http://colah.github.io/posts/2015-08-Understanding-LSTMs/):    
  ![](Images/22.png)
- Some variants on LSTM includes:
  - LSTM with **peephole connections**.
    - The normal LSTM with C<sup>\<t-1></sup> included with every gate.
- There isn't a universal superior between LSTM and it's variants. One of the advantages of GRU is that it's simpler and can be used to build much bigger network but the LSTM is more powerful and general.

### Bidirectional RNN
- There are still some ideas to let you build much more powerful sequence models. One of them is bidirectional RNNs and another is Deep RNNs.
- As we saw before, here is an example of the Name entity recognition task:  
  ![](Images/23.png)
- The name **Teddy** cannot be learned from **He** and **said**, but can be learned from **bears**.
- BiRNNs fixes this issue.
- Here is BRNNs architecture:   
  ![](Images/24.png)
- Note, that BiRNN is an **acyclic graph**.
- Part of the forward propagation goes from left to right, and part - from right to left. It learns from both sides.
- To make predictions we use y&#770;<sup>\<t></sup> by using the two activations that come from left and right.
- The blocks here can be any RNN block including the basic RNNs, LSTMs, or GRUs.
- For a lot of NLP or text processing problems, a BiRNN with LSTM appears to be commonly used.
- The disadvantage of BiRNNs that you need the entire sequence before you can process it. For example, in live speech recognition if you use BiRNNs you will need to wait for the person who speaks to stop to take the entire sequence and then make your predictions.

### Deep RNNs
- In a lot of cases the standard one layer RNNs will solve your problem. But in some problems its useful to stack some RNN layers to make a deeper network.
- For example, a deep RNN with 3 layers would look like this:  
  ![](Images/25.png)
- In feed-forward deep nets, there could be 100 or even 200 layers. In deep RNNs stacking 3 layers is already considered deep and expensive to train.
- In some cases you might see some feed-forward network layers connected after recurrent cell.


### Back propagation with RNNs
- > In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes care of the backward pass, so most deep learning engineers do not need to bother with the details of the backward pass. If however you are an expert in calculus and want to see the details of backprop in RNNs, you can work through this optional portion of the notebook.

- The quote is taken from this [notebook](https://www.coursera.org/learn/nlp-sequence-models/notebook/X20PE/building-a-recurrent-neural-network-step-by-step). If you want the details of the back propagation with programming notes look at the linked notebook.
