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

What if the pseudosentence length does not divide the length of the text (in
terms of tokens)? I chose to discard the rest of the text for the sake of
simplicity. I believe that these last few tokens will not affect the algorithm's
outcome significantly.

Should I calculate a similarity score for every _pseudosentence gap_, as Hearst
calls it? I chose not to. I only calculate the similarity scores for those gaps
where both adjacent blocks _fit_ entirely on the list of pseudosentences.

How do I determine the number of topics in a given document?  For data more
natural than Choi's, we would have to construct a heuristic that can estimate
the number of topics there are. However, our data is synthetic, and we know there
are 10 topics (segments). I still could have devised a heuristic, but I decided
to go with the simple approach.

What if I don't get all 10 topics because the block size is too big? I won't do
anything about it. The data is separated into 4 sections, which are based on
segment size. Therefore, I will expect to see effects of this decision there
(e.g. the 9-11 data producing better metric scores compared to the 3-11 data).

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
