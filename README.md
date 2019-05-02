# Topic Segmentation

Final project for NLP class at UT Austin 2019.

## Implementation of TextTiling (Hearst 1994)

Although there are already implementations of TextTiling out there
(by NLTK and other indpendents), I do it again in order to understand
the algorithm. Also, I experiment with some parts of the algorithm.
Reynar did some experimentation and found out that using depth scores was not
as accurate as using the minimum similarity scores on their test corpus (Reynar
1998). This version of TextTiling will use some of the discoveries from Reynar.

### Design Decisions

The first question is what if the pseudosentence length does not divide the
length of the text (in terms of tokens). I chose to discard the rest of the text
for the sake of simplicity. I believe that these last few tokens will not affect
the algorithm's outcome significantly.

The second is the similar question, except for blocks. Should I calculate a
similarity score for every _pseudosentence gap_, as Hearst calls it? I chose not
to. I only calculate the similarity scores for those gaps where the entire block
_fits_ on the list of pseudosentences.

## Modification: ELMo

Hearst uses a similarity score to that is essentially bag of words with cosine
similarity. Instead, I calculate the similarity score with pretrained
ELMo embeddings. ELMo gives us the contextualized word embeddings (Peters 2018).
I am not sure if ELMo will better the results here, but it seems to have helped
everything else it was applied to.

## Evaluation

### Metrics

The two metrics used in the literature are _P<sub>k</sub>_ and WinDiff, with
WinDiff being the harsher metric. I test TextTiling and its
modifications with these two metrics.

### Data

I use the Choi 2000 dataset. I thank __Freddy Choi__ for creating this data and
GitHub user __logological__ for making it easy to access through this
[repo](https://github.com/logological/C99.git).

Make sure to clone the above repo into the parent directory this repo.
The code that reads the data depends on it.
