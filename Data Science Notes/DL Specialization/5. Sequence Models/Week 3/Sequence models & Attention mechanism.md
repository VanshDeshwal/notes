
> Sequence models can be augmented using an attention mechanism. This algorithm will help your model understand where it should focus its attention given a sequence of inputs. This week, you will also learn about speech recognition and how to deal with audio data.

## Various sequence to sequence architectures

### Basic Models

- In this section we will learn about sequence to sequence - _Many to Many_ - models which are useful in various applications including machine translation and speech recognition.
- Let's start with the basic model:
    - Given this machine translation problem in which X is a French sequence and Y is an English sequence.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/52.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/52.png)
    - Our architecture will include **encoder** and **decoder**.
    - The encoder is RNN - LSTM or GRU are included - and takes the input sequence and then outputs a vector that should represent the whole input.
    - After that the decoder network, also RNN, takes the sequence built by the encoder and outputs the new sequence.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/53.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/53.png)
    - These ideas are from the following papers:
        - [Sutskever et al., 2014. Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215)
        - [Cho et al., 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)
- An architecture similar to the mentioned above works for image captioning problem:
    - In this problem X is an image, while Y is a sentence (caption).
    - The model architecture image:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/54.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/54.png)
    - The architecture uses a pretrained CNN (like AlexNet) as an encoder for the image, and the decoder is an RNN.
    - Ideas are from the following papers (they share similar ideas):
        - [Maoet et. al., 2014. Deep captioning with multimodal recurrent neural networks](https://arxiv.org/abs/1412.6632)
        - [Vinyals et. al., 2014. Show and tell: Neural image caption generator](https://arxiv.org/abs/1411.4555)
        - [Karpathy and Li, 2015. Deep visual-semantic alignments for generating image descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

### Picking the most likely sentence

- There are some similarities between the language model we have learned previously, and the machine translation model we have just discussed, but there are some differences as well.
- The language model we have learned is very similar to the decoder part of the machine translation model, except for a<0>  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/55.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/55.png)
- Problems formulations also are different:
    - In language model: $P(y^{<1>}, \dots , y^{<Ty>})$
    - In machine translation: $P(y^{<1>}, \dots, y^{<Ty>} | x^{<1>}, ..., x^{<Tx>})$
- What we don't want in machine translation model, is not to sample the output at random. This may provide some choices as an output. Sometimes you may sample a bad output.
    - Example:
        - X = "Jane visite l’Afrique en septembre."
        - Y may be:
            - Jane is visiting Africa in September.
            - Jane is going to be visiting Africa in September.
            - In September, Jane will visit Africa.
- So we need to get the best output it can be:  
- $$
\arg\max_{y^{<1>}, \ldots, y^{<T_y>}} P(y^{<1>}, \ldots, y^{<T_y>} \mid x)
$$
- The most common algorithm is the beam search, which we will explain in the next section.
- Why not use greedy search? Why not get the best choices each time?
    - It turns out that this approach doesn't really work!
    - Lets explain it with an example:
        - The best output for the example we talked about is "Jane is visiting Africa in September."
        - Suppose that when you are choosing with greedy approach, the first two words were "Jane is", the word that may come after that will be "going" as "going" is the most common word that comes after " is" so the result may look like this: "Jane is going to be visiting Africa in September.". And that isn't the best/optimal solution.
- So what is better than greedy approach, is to get an approximate solution, that will try to maximize the output (the last equation above).

### Beam Search

- Beam search is the most widely used algorithm to get the best output sequence. It's a heuristic search algorithm.
- To illustrate the algorithm we will stick with the example from the previous section. We need Y = "Jane is visiting Africa in September."
- The algorithm has a parameter `B` which is the beam width. Lets take `B = 3` which means the algorithm will get 3 outputs at a time.
- For the first step you will get ["in", "jane", "september"] words that are the best candidates.
- Then for each word in the first output, get B next (second) words and select top best B combinations where the best are those what give the highest value of multiplying both probabilities - $P(y^{<1>}|x) * P(y^{<2>}|x,y^{<1>})$. Se we will have then ["in september", "jane is", "jane visit"]. Notice, that we automatically discard _september_ as a first word.
- Repeat the same process and get the best B words for ["september", "is", "visit"] and so on.
- In this algorithm, keep only B instances of your network.
- If `B = 1` this will become the greedy search.

### Refinements to Beam Search

- In the previous section, we have discussed the basic beam search. In this section, we will try to do some refinements to it.
- The first thing is **Length optimization**
    - In beam search we are trying to optimize:  $$
\arg\max_{y^{<1>}, \ldots, y^{<T_y>}} P(y^{<1>}, \ldots, y^{<T_y>} \mid x)
$$

    - And to do that we multiply:  
        $P(y^{<1>} | x) * P(y^{<2>} | x, y^{<1>}) * \dots * P(y^{<t>} | x, y^{<y(t-1)>})$
    - Each probability is a fraction, most of the time a small fraction.
    - Multiplying small fractions will cause a **numerical overflow**. Meaning that it's too small for the floating part representation in your computer to store accurately.
    - So in practice we use **summing logs of probabilities** instead of multiplying directly. $$
\arg\max_{y} \sum_{t=1}^{T_y} \log P\!\big(y^{<t>} \mid x, y^{<1>}, \ldots, y^{<t-1>}\big)
$$
    - But there's another problem. The two optimization functions we have mentioned are preferring small sequences rather than long ones. Because multiplying more fractions gives a smaller value, so fewer fractions - bigger result.
    - So there's another step - dividing by the number of elements in the sequence.  
        $$
\frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P\!\big(y^{<t>} \mid x, y^{<1>}, \ldots, y^{<t-1>}\big)
$$

        - alpha is a hyperparameter to tune.
        - If alpha = 0 - no sequence length normalization.
        - If alpha = 1 - full sequence length normalization.
        - In practice alpha = 0.7 is a good thing (somewhere in between two extremes).
- The second thing is how can we choose best `B`?
    - The larger B - the larger possibilities, the better are the results. But it will be more computationally expensive.
    - In practice, you might see in the production setting `B=10`
    - `B=100`, `B=1000` are uncommon (sometimes used in research settings)
    - Unlike exact search algorithms like BFS (Breadth First Search) or DFS (Depth First Search), Beam Search runs faster but is not guaranteed to find the exact solution.

### Error analysis in beam search

- We have talked before on **Error analysis** in _"Structuring Machine Learning Projects"_ course. We will apply these concepts to improve our beam search algorithm.
- We will use error analysis to figure out if the `B` hyperparameter of the beam search is the problem (it doesn't get an optimal solution) or in our RNN part.
- Let's take an example:
    - Initial info:
        - x = "Jane visite l’Afrique en septembre."
        - y* = "Jane visits Africa in September." - right answer
        - ŷ = "Jane visited Africa last September." - answer produced by model
    - Our model that has produced not a good result.
    - We now want to know who to blame - the RNN or the beam search.
    - To do that, we calculate P(y* | X) and P(ŷ | X). There are two cases:
        - Case 1 (P(y* | X) > P(ŷ | X)):
            - Conclusion: Beam search is at fault.
        - Case 2 (P(y* | X) <= P(ŷ | X)):
            - Conclusion: RNN model is at fault.
- The error analysis process is as following:
    - You choose N error examples and make the following table:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/59.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/59.png)
    - `B` for beam search, `R` is for the RNN.
    - Get counts and decide what to work on next.

### BLEU Score

- One of the challenges of machine translation, is that given a sentence in a language there are one or more possible good translation in another language. So how do we evaluate our results?
- The way we do this is by using **BLEU score**. BLEU stands for _bilingual evaluation understudy_.
- The intuition is: as long as the machine-generated translation is pretty close to any of the references provided by humans, then it will get a high BLEU score.
- Let's take an example:
    - X = "Le chat est sur le tapis."
    - Y1 = "The cat is on the mat." (human reference 1)
    - Y2 = "There is a cat on the mat." (human reference 2)
    - Suppose that the machine outputs: "the the the the the the the."
    - One way to evaluate the machine output is to look at each word in the output and check if it is in the references. This is called _precision_:
        - precision = 7/7 because "the" appeared in Y1 or Y2
    - This is not a useful measure!
    - We can use a modified precision in which we are looking for the reference with the maximum number of a particular word and set the maximum appearing of this word to this number. So:
        - modified precision = 2/7 because the max is 2 in Y1
        - We clipped the 7 times by the max which is 2.
    - Here we are looking at one word at a time - unigrams, we may look at n-grams too
- BLEU score on bigrams
    - The **n-grams** typically are collected from a text or speech corpus. When the items are words, **n-grams** may also be called shingles. An **n-gram** of size 1 is referred to as a "unigram"; size 2 is a "bigram" (or, less commonly, a "digram"); size 3 is a "trigram".
        
    - X = "Le chat est sur le tapis."
        
    - Y1 = "The cat is on the mat."
        
    - Y2 = "There is a cat on the mat."
        
    - Suppose that the machine outputs: "the cat the cat on the mat."
        
    - The bigrams in the machine output:
        
| Pairs      | Count | Count clip |
| ---------- | ----- | ---------- |
| the cat    | 2     | 1 (Y1)     |
| cat the    | 1     | 0          |
| cat on     | 1     | 1 (Y2)     |
| on the     | 1     | 1 (Y1)     |
| the mat    | 1     | 1 (Y1)     |
| **Totals** | 6     | 4          |
        
        Modified precision = sum(Count clip) / sum(Count) = 4/6
        
- So here are the equations for modified precision for the n-grams case:
- $$
p_1 = \frac{\displaystyle \sum_{\text{unigram} \in \hat{y}} count_{clip}(\text{unigram})}
           {\displaystyle \sum_{\text{unigram} \in \hat{y}} count(\text{unigram})}
\qquad\qquad
p_n = \frac{\displaystyle \sum_{\text{ngram} \in \hat{y}} count_{clip}(\text{ngram})}
           {\displaystyle \sum_{\text{ngram} \in \hat{y}} count(\text{ngram})}
$$

- Let's put this together to formalize the BLEU score:
    - **Pn** = Bleu score on one type of n-gram
    - **Combined BLEU score** = BP * exp(1/n * sum(Pn))
        - For example if we want BLEU for 4, we compute P1, P2, P3, P4 and then average them and take the exp.
    - **BP** is called **BP penalty** which stands for brevity penalty. It turns out that if a machine outputs a small number of words it will get a better score so we need to handle that.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/62.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/62.png)
- BLEU score has several open source implementations.
- It is used in a variety of systems like machine translation and image captioning.

### Attention Model Intuition

- So far we were using sequence to sequence models with an encoder and decoders. There is a technique called _attention_ which makes these models even better.
- The attention idea has been one of the most influential ideas in deep learning.
- The problem of long sequences:
    - Given this model, inputs, and outputs.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/63.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/63.png)
    - The encoder should memorize this long sequence into one vector, and the decoder has to process this vector to generate the translation.
    - If a human would translate this sentence, he/she wouldn't read the whole sentence and memorize it then try to translate it. He/she translates a part at a time.
    - The performance of this model decreases if a sentence is long.
    - We will discuss the attention model that works like a human that looks at parts at a time. That will significantly increase the accuracy even with longer sequence:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/64.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/64.png)
        - Blue is the normal model, while green is the model with attention mechanism.
- In this section we will give just some intuitions about the attention model and in the next section we will discuss it's details.
- At first the attention model was developed for machine translation but then other applications used it like computer vision and new architectures like Neural Turing machine.
- The attention model was descried in this paper:
    - [Bahdanau et. al., 2014. Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
- Now for the intuition:
    - Suppose that our encoder is a bidirectional RNN: [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/65.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/65.png)
    - We give the French sentence to the encoder and it should generate a vector that represents the inputs.
    - Now to generate the first word in English which is "Jane" we will make another RNN which is the decoder.
    - Attention weights are used to specify which words are needed when to generate a word. So to generate "jane" we will look at "jane", "visite", "l'Afrique"  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/66.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/66.png)
    - alpha<1,1>, alpha<1,2>, and alpha<1,3> are the attention weights being used.
    - And so to generate any word there will be a set of attention weights that controls which words we are looking at right now.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/67.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/67.png)

### Attention Model

- Lets formalize the intuition from the last section into the exact details on how this can be implemented.
- First we will have an bidirectional RNN (most common is LSTMs) that encodes French language:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/68.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/68.png)
- For learning purposes, lets assume that a<t'> will include the both directions activations at time step t'.
- We will have a unidirectional RNN to produce the output using a context `c` which is computed using the attention weights, which denote how much information does the output needs to look in a<t'>  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/69.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/69.png)
- Sum of the attention weights for each element in the sequence should be 1:  $$
\sum_{t'} \alpha^{\langle 1, t' \rangle} = 1
$$

- The context `c` is calculated using this equation:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/71.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/71.png)
- Lets see how can we compute the attention weights:
    - So $\alpha^{<t, t'>}$ = amount of attention $y^{<t>}$ should pay to $a^{<t'>}$
        - Like for example we payed attention to the first three words through $\alpha^{<1,1>}$, $\alpha^{<1,2>}$, $\alpha^{<1,3>}$
    - We are going to softmax the attention weights so that their sum is 1:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/72.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/72.png)
    - Now we need to know how to calculate e<t, t'>. We will compute e using a small neural network (usually 1-layer, because we will need to compute this a lot):  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/73.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/73.png)
        - s<t-1> is the hidden state of the RNN s, and a<t'> is the activation of the other bidirectional RNN.
- One of the disadvantages of this algorithm is that it takes quadratic time or quadratic cost to run.
- One fun way to see how attention works is by visualizing the attention weights:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/74.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/74.png)

### Speech recognition - Audio data

[](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Readme.md#speech-recognition---audio-data)

#### Speech recognition

[](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Readme.md#speech-recognition)

- One of the most exciting developments using sequence-to-sequence models has been the rise of very accurate speech recognition.
- Let's define the speech recognition problem:
    - X: audio clip
    - Y: transcript
    - If you plot an audio clip it will look like this:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/75.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/75.png)
        - The horizontal axis is time while the vertical is changes in air pressure.
    - What really is an audio recording? A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear perceives as sound. You can think of an audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone. We will use audio sampled at 44100 Hz (or 44100 Hertz). This means the microphone gives us 44100 numbers per second. Thus, a 10 second audio clip is represented by 441000 numbers (= 10 * 44100).
    - It is quite difficult to work with "raw" representation of audio.
    - Because even human ear doesn't process raw wave forms, the human ear can process different frequencies.
    - There's a common preprocessing step for an audio - generate a spectrogram which works similarly to human ears.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/76.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/76.png)
        - The horizontal axis is time while the vertical is frequencies. Intensity of different colors shows the amount of energy - how loud is the sound for different frequencies (a human ear does a very similar preprocessing step).
    - A spectrogram is computed by sliding a window over the raw audio signal, and calculates the most active frequencies in each window using a Fourier transformation.
    - In the past days, speech recognition systems were built using _phonemes_ that are a hand engineered basic units of sound. Linguists used to hypothesize that writing down audio in terms of these basic units of sound called _phonemes_ would be the best way to do speech recognition.
    - End-to-end deep learning found that phonemes was no longer needed. One of the things that made this possible is the large audio datasets.
    - Research papers have around 300 - 3000 hours of training data while the best commercial systems are now trained on over 100,000 hours of audio.
- You can build an accurate speech recognition system using the attention model that we have descried in the previous section:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/77.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/77.png)
- One of the methods that seem to work well is _CTC cost_ which stands for "Connectionist temporal classification"
    - To explain this let's say that Y = "the quick brown fox"
    - We are going to use an RNN with input, output structure:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/78.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/78.png)
    - Note: this is a unidirectional RNN, but in practice a bidirectional RNN is used.
    - Notice, that the number of inputs and number of outputs are the same here, but in speech recognition problem input X tends to be a lot larger than output Y.
        - 10 seconds of audio at 100Hz gives us X with shape (1000, ). These 10 seconds don't contain 1000 character outputs.
    - The CTC cost function allows the RNN to output something like this:
        - `ttt_h_eee<SPC>___<SPC>qqq___` - this covers "the q".
        - The _ is a special character called "blank" and `<SPC>` is for the "space" character.
        - Basic rule for CTC: collapse repeated characters not separated by "blank"
    - So the 19 character in our Y can be generated into 1000 character output using CTC and it's special blanks.
    - The ideas were taken from this paper:
        - [Graves et al., 2006. Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks](https://dl.acm.org/citation.cfm?id=1143891)
        - This paper's ideas were also used by Baidu's DeepSpeech.
- Using both attention model and CTC cost can help you to build an accurate speech recognition system.

#### Trigger Word Detection

[](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Readme.md#trigger-word-detection)

- With the rise of deep learning speech recognition, there are a lot of devices that can be waked up by saying some words with your voice. These systems are called trigger word detection systems.
- For example, Alexa - a smart device made by Amazon - can answer your call "Alexa, what time is it?" and then Alexa will respond to you.
- Trigger word detection systems include:  
    [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/79.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/79.png)
- For now, the trigger word detection literature is still evolving so there actually isn't a single universally agreed on the algorithm for trigger word detection yet. But let's discuss an algorithm that can be used.
- Let's now build a model that can solve this problem:
    - X: audio clip
    - X has been preprocessed and spectrogram features have been returned of X
        - X<1>, X<2>, ... , X<t>
    - Y will be labels 0 or 1. 0 represents the non-trigger word, while 1 is that trigger word that we need to detect.
    - The model architecture can be like this:  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/80.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/80.png)
        - The vertical lines in the audio clip represent moment just after the trigger word. The corresponding to this will be 1.
    - One disadvantage of this creates a very imbalanced training set. There will be a lot of zeros and few ones.
    - A hack to solve this is to make an output a few ones for several times or for a fixed period of time before reverting back to zero.  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/81.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/81.png)  
        [![](https://github.com/amanchadha/coursera-deep-learning-specialization/raw/master/C5%20-%20Sequence%20Models/Notes/Images/85.png)](https://github.com/amanchadha/coursera-deep-learning-specialization/blob/master/C5%20-%20Sequence%20Models/Notes/Images/85.png)

## Extras