
> Natural language processing with deep learning is an important combination. Using word vector representations and embedding layers you can train recurrent neural networks with outstanding performances in a wide variety of industries. Examples of applications are sentiment analysis, named entity recognition and machine translation.

## Introduction to Word Embeddings

### Word Representation

- NLP has been revolutionized by deep learning and especially by RNNs and deep RNNs.
- Word embeddings is a way of representing words. It lets your algorithm automatically understand the analogies between words like "king" and "queen".
- So far we have defined our language by a vocabulary. Then represented our words with a one-hot vector that represents the word in the vocabulary.
    - An image example would be:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/27.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/27.png)
    - We will use the annotation **O**$_{idx}$ for any word that is represented with one-hot like in the image.
    - One of the weaknesses of this representation is that it treats a word as a thing that itself and it doesn't allow an algorithm to generalize across words.
        - For example: "I want a glass of **orange** $\underline{\hspace{2cm}}$ ", a model should predict the next word as **juice**.
        - A similar example "I want a glass of **apple**  $\underline{\hspace{2cm}}$ ", a model won't easily predict **juice** here if it wasn't trained on it. And if so the two examples aren't related although orange and apple are similar.
    - Inner product between any one-hot encoding vector is zero. Also, the distances between them are the same.
- So, instead of a one-hot presentation, won't it be nice if we can learn a featurized representation with each of these words: man, woman, king, queen, apple, and orange?  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/28.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/28.png)  - Each word will have a, for example, 300 features with a type of float point number.
    - Each word column will be a 300-dimensional vector which will be the representation.
    - We will use the notation **e**5391 to describe **man** word features vector.
    - Now, if we return to the examples we described again:
        - "I want a glass of **orange**  $\underline{\hspace{2cm}}$ "
        - I want a glass of **apple**  $\underline{\hspace{2cm}}$ 
    - Orange and apple now share a lot of similar features which makes it easier for an algorithm to generalize between them.
    - We call this representation **Word embeddings**.
- To visualize word embeddings we use a t-SNE algorithm to reduce the features to 2 dimensions which makes it easy to visualize:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/29.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/29.png)
    - You will get a sense that more related words are closer to each other.
- The **word embeddings** came from that we need to embed a unique vector inside a n-dimensional space.

### Using word embeddings

- Let's see how we can take the feature representation we have extracted from each word and apply it in the Named entity recognition problem.
- Given this example (from named entity recognition):  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/30.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/30.png)
- **Sally Johnson** is a person's name.
- After training on this sentence the model should find out that the sentence "**Robert Lin** is an _apple_ farmer" contains Robert Lin as a name, as apple and orange have near representations.
- Now if you have tested your model with this sentence "**Mahmoud Badry** is a _durian_ cultivator" the network should learn the name even if it hasn't seen the word _durian_ before (during training). That's the power of word representations.
- The algorithms that are used to learn **word embeddings** can examine billions of words of unlabeled text - for example, 100 billion words and learn the representation from them.
- Transfer learning and word embeddings:
    1. Learn word embeddings from large text corpus (1-100 billion of words).
        - Or download pre-trained embedding online.
    2. Transfer embedding to new task with the smaller training set (say, 100k words).
    3. Optional: continue to finetune the word embeddings with new data.
        - You bother doing this if your smaller training set (from step 2) is big enough.
- Word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller training set.
- Also, one of the advantages of using word embeddings is that it reduces the size of the input!
    - 10,000 one hot compared to 300 features vector.
- Word embeddings have an interesting relationship to the face recognition task:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/31.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/31.png)
    - In this problem, we encode each face into a vector and then check how similar are these vectors.
    - Words **encoding** and **embeddings** have a similar meaning here.
- In the word embeddings task, we are learning a representation for each word in our vocabulary (unlike in image encoding where we have to map each new image to some n-dimensional vector). We will discuss the algorithm in next sections.

### Properties of word embeddings

- One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. While analogy reasoning may not be by itself the most important NLP application, but it might help convey a sense of what these word embeddings can do.
- Analogies example:
    - Given this word embeddings table:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/32.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/32.png)
    - Can we conclude this relation:
        - Man ==> Woman
        - King ==> ??
    - Lets subtract e$_{Man}$ from e$_{Woman}$. This will equal the vector `[-2 0 0 0]`
    - Similar e$_{King}$ - e$_{Queen}$ = `[-2 0 0 0]`
    - So the difference is about the gender in both.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/33.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/33.png)
        - This vector represents the gender.
        - This drawing is a 2D visualization of the 4D vector that has been extracted by a t-SNE algorithm. It's a drawing just for visualization. Don't rely on the t-SNE algorithm for finding parallels.
    - So we can reformulate the problem to find:
        - e$_{Man}$ - e$_{Woman}$ ≈ e$_{King}$ - e??
    - It can also be represented mathematically by:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/34.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/34.png)
    - It turns out that e$_{Queen}$ is the best solution here that gets the the similar vector.
- Cosine similarity - the most commonly used similarity function:
    - Equation:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/35.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/35.png)
        - `CosineSimilarity(u, v)` = `u . v` / `||u|| ||v||` = cos(θ)
        - The top part represents the inner product of `u` and `v` vectors. It will be large if the vectors are very similar.
- You can also use Euclidean distance as a similarity function (but it rather measures a dissimilarity, so you should take it with negative sign).
- We can use this equation to calculate the similarities between word embeddings and on the analogy problem where `u` = e$_w$ and `v` = e$_{king}$ - e$_{man}$ + e$_{woman}$

### Embedding matrix

- When you implement an algorithm to learn a word embedding, what you end up learning is a **embedding matrix**.
- Let's take an example:
    - Suppose we are using 10,000 words as our vocabulary (plus token).
    - The algorithm should create a matrix `E` of the shape (300, 10000) in case we are extracting 300 features.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/36.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/36.png)
    - If $O_{6257}$ is the one hot encoding of the word **orange** of shape (10000, 1), then  
        _np.dot(`E`,$O_{6257}$) = $e_{6257}$_ which shape is (300, 1).
    - Generally _np.dot(`E`, $O_j$) = $e_j$_
- In the next sections, you will see that we first initialize `E` randomly and then try to learn all the parameters of this matrix.
- In practice it's not efficient to use a dot multiplication when you are trying to extract the embeddings of a specific word, instead, we will use slicing to slice a specific column. In Keras there is an embedding layer that extracts this column with no multiplication.

## Learning Word Embeddings: Word2vec & GloVe

### Learning word embeddings

- Let's start learning some algorithms that can learn word embeddings.
- At the start, word embeddings algorithms were complex but then they got simpler and simpler.
- We will start by learning the complex examples to make more intuition.
- **Neural language model**:
    - Let's start with an example:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/37.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/37.png)
    - We want to build a language model so that we can predict the next word.
    - So we use this neural network to learn the language model  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/38.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/38.png)
        - We get $e_j$ by np.dot(E,$O_j$)
        - NN layer has parameters `W1` and `b1` while softmax layer has parameters `W2` and `b2`
        - Input dimension is (300x6, 1) if the window size is 6 (six previous words).
        - Here we are optimizing `E` matrix and layers parameters. We need to maximize the likelihood to predict the next word given the context (previous words).
    - This model was build in 2003 and tends to work pretty decent for learning word embeddings.
- In the last example we took a window of 6 words that fall behind the word that we want to predict. There are other choices when we are trying to learn word embeddings.
    - Suppose we have an example: "I want a glass of orange **juice** to go along with my cereal"
    - To learn **juice**, choices of **context** are:
        1. Last 4 words.
            - We use a window of last 4 words (4 is a hyperparameter), "a glass of orange" and try to predict the next word from it.
        2. 4 words on the left and on the right.
            - "a glass of orange" and "to go along with"
        3. Last 1 word.
            - "orange"
        4. Nearby 1 word.
            - "glass" word is near juice.
            - This is the idea of **skip grams** model.
            - The idea is much simpler and works remarkably well.
            - We will talk about this in the next section.
- Researchers found that if you really want to build a _language model_, it's natural to use the last few words as a context. But if your main goal is really to learn a _word embedding_, then you can use all of these other contexts and they will result in very meaningful work embeddings as well.
- To summarize, the language modeling problem poses a machines learning problem where you input the context (like the last four words) and predict some target words. And posing that problem allows you to learn good word embeddings.

### Word2Vec

- Before presenting Word2Vec, lets talk about **skip-grams**:
    - For example, we have the sentence: "I want a glass of orange juice to go along with my cereal"
        
    - We will choose **context** and **target**.
        
    - The target is chosen randomly based on a window with a specific size.
        
| Context | Target | How far |
| ------- | ------ | ------- |
| orange  | juice  | +1      |
| orange  | glass  | -2      |
| orange  | my     | +6      |

- We have converted the problem into a supervised problem.
        
    - This is not an easy learning problem because learning within -10/+10 words (10 - an example) is hard.
        
    - We want to learn this to get our word embeddings model.
        
- Word2Vec model:
    - Vocabulary size = 10,000 words
    - Let's say that the context word are `c` and the target word is `t`
    - We want to learn `c` to `t`
    - We get ec by `E`. $O_c$
    - We then use a softmax layer to get `P(t|c)` which is ŷ
    - Also we will use the cross-entropy loss function.
    - This model is called skip-grams model.
- The last model has a problem with the softmax layer:  
$$
p(t \mid c) = \frac{e^{\theta_t^T e_c}}{\sum_{j=1}^{10,000} e^{\theta_j^T e_c}}

$$
    - Here we are summing 10,000 numbers which corresponds to the number of words in our vocabulary.
    - If this number is larger say 1 million, the computation will become very slow.
- One of the solutions for the last problem is to use "**Hierarchical softmax classifier**" which works as a tree classifier.  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/40.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/40.png)
- In practice, the hierarchical softmax classifier doesn't use a balanced tree like the drawn one. Common words are at the top and less common are at the bottom.
- How to sample the context **c**?
    - One way is to choose the context by random from your corpus.
    - If you have done it that way, there will be frequent words like "the, of, a, and, to, .." that can dominate other words like "orange, apple, durian,..."
    - In practice, we don't take the context uniformly random, instead there are some heuristics to balance the common words and the non-common words.
- word2vec paper includes 2 ideas of learning word embeddings. One is skip-gram model and another is CBoW (continuous bag-of-words).

### Negative Sampling

- Negative sampling allows you to do something similar to the skip-gram model, but with a much more efficient learning algorithm. We will create a different learning problem.
    
- Given this example:
    
    - "I want a glass of orange juice to go along with my cereal"
- The sampling will look like this:
    
| Context | Word  | target |
| ------- | ----- | ------ |
| orange  | juice | 1      |
| orange  | king  | 0      |
| orange  | book  | 0      |
| orange  | the   | 0      |
| orange  | of    | 0      |
    
We get positive example by using the same skip-grams technique, with a fixed window that goes around.
    
- To generate a negative example, we pick a word randomly from the vocabulary.
    
- Notice, that we got word "of" as a negative example although it appeared in the same sentence.
    
- So the steps to generate the samples are:
    
    1. Pick a positive context
    2. Pick a k negative contexts from the dictionary.
- k is recommended to be from 5 to 20 in small datasets. For larger ones - 2 to 5.
    
- We will have a ratio of k negative examples to 1 positive ones in the data we are collecting.
    
- Now let's define the model that will learn this supervised learning problem:
    
    - Lets say that the context word are `c` and the word are `t` and `y` is the target.
    - We will apply the simple logistic regression model. 
    - $$
     P(y = 1 \mid c, t) = \sigma(\theta_t^T e_c)

     $$
    - The logistic regression model can be drawn like this:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/42.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/42.png)
    - So we are like having 10,000 binary classification problems, and we only train k+1 classifier of them in each iteration.
- How to select negative samples:
    
    - We can sample according to empirical frequencies in words corpus which means according to how often different words appears. But the problem with that is that we will have more frequent words like _the, of, and..._
    - The best is to sample with this equation (according to authors): 
    $$
    P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=1}^{10,000} f(w_j)^{3/4}}

    $$

### GloVe word vectors

- GloVe is another algorithm for learning the word embedding. It's the simplest of them.
    
- This is not used as much as word2vec or skip-gram models, but it has some enthusiasts because of its simplicity.
    
- GloVe stands for Global vectors for word representation.
    
- Let's use our previous example: "I want a glass of orange juice to go along with my cereal".
    
- We will choose a context and a target from the choices we have mentioned in the previous sections.
    
- Then we will calculate this for every pair: $X_{ct}$ = # times `t` appears in context of `c`
    
- $X_{ct}$= $X_{tc}$ if we choose a window pair, but they will not equal if we choose the previous words for example. In GloVe they use a window which means they are equal
    
- The model is defined like this:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/44.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/44.png)
    
- f(x) - the weighting term, used for many reasons which include:
    
    - The `log(0)` problem, which might occur if there are no pairs for the given target and context values.
    - Giving not too much weight for stop words like "is", "the", and "this" which occur many times.
    - Giving not too little weight for infrequent words.
- **Theta** and **e** are symmetric which helps getting the final word embedding.
    
- _Conclusions on word embeddings:_
    
    - If this is your first try, you should try to download a pre-trained model that has been made and actually works best.
    - If you have enough data, you can try to implement one of the available algorithms.
    - Because word embeddings are very computationally expensive to train, most ML practitioners will load a pre-trained set of embeddings.
    - A final note that you can't guarantee that the axis used to represent the features will be well-aligned with what might be easily humanly interpretable axis like gender, royal, age.

## Applications using Word Embeddings

### Sentiment Classification

- As we have discussed before, Sentiment classification is the process of finding if a text has a positive or a negative review. Its so useful in NLP and is used in so many applications. An example would be:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/45.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/45.png)
- One of the challenges with it, is that you might not have a huge labeled training data for it, but using word embeddings can help getting rid of this.
- The common dataset sizes varies from 10,000 to 100,000 words.
- A simple sentiment classification model would be like this:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/46.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/46.png)
    - The embedding matrix may have been trained on say 100 billion words.
    - Number of features in word embedding is 300.
    - We can use **sum** or **average** given all the words then pass it to a softmax classifier. That makes this classifier works for short or long sentences.
- One of the problems with this simple model is that it ignores words order. For example "Completely lacking in **good** taste, **good** service, and **good** ambience" has the word _good_ 3 times but its a negative review.
- A better model uses an RNN for solving this problem:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/47.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/47.png)
    - And so if you train this algorithm, you end up with a pretty decent sentiment classification algorithm.
    - Also, it will generalize better even if words weren't in your dataset. For example you have the sentence "Completely **absent** of good taste, good service, and good ambience", then even if the word "absent" is not in your label training set, if it was in your 1 billion or 100 billion word corpus used to train the word embeddings, it might still get this right and generalize much better even to words that were in the training set used to train the word embeddings but not necessarily in the label training set that you had for specifically the sentiment classification problem.

### Debiasing word embeddings

- We want to make sure that our word embeddings are free from undesirable forms of bias, such as gender bias, ethnicity bias and so on.
- Horrifying results on the trained word embeddings in the context of Analogies:
    - Man : Computer_programmer as Woman : **Homemaker**
    - Father : Doctor as Mother : **Nurse**
- Word embeddings can reflect gender, ethnicity, age, sexual orientation, and other biases of text used to train the model.
- Learning algorithms by general are making important decisions and it mustn't be biased.
- Andrew thinks we actually have better ideas for quickly reducing the bias in AI than for quickly reducing the bias in the human race, although it still needs a lot of work to be done.
- Addressing bias in word embeddings steps:
    - Idea from the paper: [https://arxiv.org/abs/1607.06520](https://arxiv.org/abs/1607.06520)
    - Given these learned embeddings:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/48.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/48.png)
    - We need to solve the **gender bias** here. The steps we will discuss can help solve any bias problem but we are focusing here on gender bias.
    - Here are the steps:
        1. Identify the direction:
            - Calculate the difference between:
                - $e_{he} - e_{she}$
                - $e_{male} - e_{female}$
                - ....
            - Choose some k differences and average them.
            - This will help you find this:  
                [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/49.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/49.png)
            - By that we have found the bias direction which is 1D vector and the non-bias vector which is 299D vector.
        2. Neutralize: For every word that is not definitional, project to get rid of bias.
            - Babysitter and doctor need to be neutral so we project them on non-bias axis with the direction of the bias:  
                [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/50.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/50.png)
                - After that they will be equal in the term of gender.         - To do this the authors of the paper trained a classifier to tell the words that need to be neutralized or not.
        3. Equalize pairs
            - We want each pair to have difference only in gender. Like:
                - Grandfather - Grandmother         - He - She         - Boy - Girl
            - We want to do this because the distance between grandfather and babysitter is bigger than babysitter and grandmother:  
                [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/51.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/51.png)
            - To do that, we move grandfather and grandmother to a point where they will be in the middle of the non-bias axis.
            - There are some words you need to do this for in your steps. Number of these words is relatively small.