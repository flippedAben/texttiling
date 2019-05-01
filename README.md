# Topic Segmentation

Final project for NLP class at UT Austin 2019.

## Implementation of TextTiling (Hearst 1994)

Although there are already implementations of TextTiling out there
(by NLTK and other indpendents), I want to do it again in order to understand
the algorithm. Also, I want to experiment with some parts of the algorithm.
Reynar did some experimentation and found out that using depth scores was not
as accurate as using the minimum similarity scores on their test corpus (Reynar
1998). This version of TextTiling will use some of the discoveries from Reynar.

## Modification: ELMo

Hearst uses a similarity score to that is essentially bag of words with cosine
similarity. Instead, I want to calculate the similarity score with pretrained
ELMo embeddings. ELMo gives us the contextualized word embeddings (Peters 2018).
I am not sure if ELMo will better the results here, but it seems to have helped
everything else it was applied to.

## Evaluation

The two metrics used in the literature are P<sub>k</sub> and WinDiff, with
WinDiff being the harsher metric. I want to put test TextTiling and its
modifications with these two metrics.

I want to use the Choi 2000 dataset.
