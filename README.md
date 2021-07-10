# Word Vectors
## Introduction
Language is an efficient way to compress and communicate information. The communication through language assumes the end users possesses a great deal of knowledge on subject matters. This assumption enables language to compress and convey information efficiently. For example, when we say to a friend: "the playground we used to go each afternoon in our childhood, is now equipped with state of the facilities". Although it only takes a few bytes to encode the sentence, it packs information that may take Gigabytes to represent in a computer e.g. the visual model of a playground, details of facilities. 

However, if the assumption is not valid, the full information can't be delivered. Therefore, for a sophisticated system like human language (natural language) to work as efficiently in computer communications, machines need to have sound knowledge of subject matters and quirks of natural communications. To this end, ML models are trained on huge corpus of text data (e.g. wikipedia) to impart the relevant understanding to computer systems. Since computer can potentially memorize way more information than humans, can the language communications among computers be more efficient than humans?

Words are building blocks of each natural language. So its imperative to find a reasonable way to represent word meanings in terms of numbers/bits.

## How do we represent meaning of a word?
A typical linguistic way of thinking meaning in terms of denotational semantics:
    * signifier (symbol) <=> signified (idea/thing)
Hence, for example the word "chair" denotes all the physical chairs out in the world, "running" denotes actions with quick movement by self. 

### NLTK
To work with word meaning, we typically use an online word dictionary. Alternatively we can use NLTK (like swiss army knife of NLP, not very good but useful for baseline)

```python
from nltk.corpus import wordnet as wn
poses = {'n': 'noun', 'v': 'verb', 's': 'adj (s)', 'a': 'adj (s)', 'r': 'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()], ", ".join([l.name() for l in synset.lemmas()])))
```
NLTK maintains a word dictionary called wordnet. But it has following disadvantages:
* Lacks fine grained meaning in context (missing nuance) 
    * e.g. "proficient" is listed as synonym for "good". However, this is only correct in some context.
* Not agile, requires manual effort to maintain. Missing new meanings of words (meaning of words can change over time)

### One-Hot Vectors
Alternatively, words can be denoted as discrete symbols. We can create a vocabulary of all words and each word gets one 1 at its index position in the vocabulary, rest are zeros e.g. 
* hotel = [0, 0, 0, 0, 1, 0, 0, 0]
* university = [0, 0, 0, 0, 0, 0, 0, 1]

This representation also has following disadvantages:
* All pair of words are orthogonal to each other i.e. no natural notion of similarity between words
* Need to rely on external libraries for similarity

### Distributional Semantics
Get meaning of word from the context they appear in frequently. 
    * You shall know a word by the company it keeps
    * Words are represented by a finite dimensional vectors
    * The dimension of vectors is significantly less than the vocab size.
    * Elements of these vectors are usually non-zeros (dense representation)
    * e.g. Banking = [0.213, 0.987, -0.001, -0.223, 0.786, 0.912]

## Word2vec
Word2vec is a framework for learning word meaning. It uses the distributed sematic concept. 

Idea:
* We have a large corpus of text
* Every word in a fixed vocabulary is represented by a vector
* Go through each position `t` in the text, which has a center word `c` and context words `o`
* Use the similarity of the word vectors for `c` and `o` to calculate 
    * the probability of `c` given `o` -> Continuous Bag of Word (CBOW)
    * the probability of `o` given `c` -> Skip-gram 
* Keep adjusting the word vectors to maximize this probability

### How to compute probability of a context word given the center word?
Let's denote the embedding of a word as
* `u_w` when word `w` is context word
* `v_w` when word `w` is center word

Now probability of concent word given the center word can be written as

<img src="https://latex.codecogs.com/svg.latex?P(o|c)&space;=&space;\frac{\text{exp}(v_{c}^{T}u_o)}{\sum_{w&space;\in&space;V}&space;\text{exp}(v_{c}^{T}u_w)}" title="P(o|c)&space;=&space;\frac{\text{exp}(v_{c}^{T}u_o)}{\sum_{w&space;\in&space;V}&space;\text{exp}(v_{c}^{T}u_w)}" />

This uses three concepts
* Dot product to capture similarity between two words 
* Exponentiation to make any real number positive
* Softmax function to convert an arbitrary vector to probability distribution 
