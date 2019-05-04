import bisect
import collections
import glob
import math
import nltk
import os
import segeval
import string

# A single sample from the Choi 2000 data
class Sample(object):
    def __init__(self):
        self.segments = []

    def add_segment(self, para):
        self.segments.append(para)

    def get_segments(self):
        # List[List[sentences]]
        return self.segments

    def get_sent_bound_idxs(self):
        # return sentence boundary indices for all segments
        result = []
        for seg in self.segments:
            if not result:
                result.append(len(seg))
            else:
                result.append(len(seg) + result[-1])
        return result


# TextTiling algorithm with modifications (not including ELMo)
class TextTiler(object):
    def __init__(self, w, k):
        self.w = w
        self.k = k

    def eval_tile_text(self, sample):
        '''
        Returns a tuple of metric scores (Pk, WinDiff, B).
        '''
        ### Record paragraph break points
        # this list maps sentence index to beginning token index
        sent_bounds = [0]
        normed_text = []
        for seg in sample.get_segments():
            normed_seg_length = 0
            for sent in seg:
                normed_sent = self.normalize_text(sent)
                normed_seg_length += len(normed_sent)
                sent_bounds.append(len(normed_sent) + sent_bounds[-1])
                normed_text += normed_sent
        del sent_bounds[-1]

        ### Break up text into Pseudosentences
        # this list maps pseudosentence index to beginning token index
        ps_bounds = list(range(0, len(normed_text), self.w))
        pseudosents = [normed_text[i:i + self.w] for i in ps_bounds]

        # discard pseudosents of length < self.w
        # also, record the waste for fun
        waste = 0
        if len(pseudosents[-1]) < self.w:
            waste += len(pseudosents[-1])
            print(f'Waste so far: {waste} tokens')
            del pseudosents[-1]

        ### Group into blocks and calculate sim scores
        # List[Tuple(sim score, pseudosent index)]
        # here, the index is of the first PS in block_b
        sims = []
        i = 0
        while i + 2 * self.k <= len(pseudosents):
            mid = i + self.k
            end = i + 2 * self.k
            block_a = pseudosents[i:mid]
            block_b = pseudosents[mid:end]

            a = [token for ps in block_a for token in ps]
            b = [token for ps in block_b for token in ps]
            bow_a = collections.Counter(a)
            bow_b = collections.Counter(b)

            sims.append(self.sim(bow_a, bow_b))
            i += 1

        ### Find boundaries (valleys)
        pred = []
        for j in range(0, len(sims)):
            if j != 0 and j != len(sims) - 1:
                if sims[j] < sims[j-1] and sims[j] < sims[j+1]:
                    pred.append(j)
            j += 1
        pred = [j + self.k for j in pred]

        ### Evalute
        # map pseudosentence indices to beginning token index
        pred_btokis = [ps_bounds[i] for i in pred]
        # map beginning token index to closest sentence index
        # (this token is closest to the beginning of which sentence?)
        pred_sentis = [self.btoki_to_senti(t, sent_bounds) for t in pred_btokis]
        # add last boundary (which we know is always there)
        pred_sentis += [len(sent_bounds)]
        gold_sentis = sample.get_sent_bound_idxs()

        pred = self.array_derivative(pred_sentis)
        gold = self.array_derivative(gold_sentis)

        pk = segeval.pk(pred, gold)
        wd = segeval.window_diff(pred, gold)
        bs = segeval.boundary_similarity(pred, gold, one_minus=True)

        return (pk, wd, bs)

    def normalize_text(self, segment):
        tokens = nltk.tokenize.word_tokenize(segment)
        tokens = [t.lower() for t in tokens]
        no_punct = str.maketrans('', '', string.punctuation)
        tokens = [t.translate(no_punct) for t in tokens]
        tokens = [t for t in tokens if t.isalpha()]

        ps = nltk.stem.porter.PorterStemmer()
        stems = [ps.stem(w) for w in tokens]

        sw = nltk.corpus.stopwords.words('english')
        sw = set(sw)
        words = [w for w in stems if w not in sw]
        return words

    def sim(self, a, b):
        union = a | b
        prod = [a[tok] * b[tok] for tok in union]
        numerator = sum(prod)

        sqsum_a = sum([a[tok]**2 for tok in a])
        sqsum_b = sum([b[tok]**2 for tok in b])
        denominator = math.sqrt(sqsum_a * sqsum_b)

        return numerator/denominator

    def btoki_to_senti(self, btoki, sent_bounds):
        l = bisect.bisect_left(sent_bounds, btoki)
        h = bisect.bisect_right(sent_bounds, btoki)
        choose_l = abs(l - btoki) < abs(h - btoki)
        return l if choose_l else h

    def array_derivative(self, a):
        result = [a[0]]
        for i in range(1, len(a)):
            result.append(a[i] - a[i-1])
        return result


def read_samples(path):
    '''
    Returns a dictionary: types -> List[Sample]
    '''
    # since I don't expect the data repo to change soon (the last commit was in
    # 2016), I will hardcode in the file names:
    #   1, 2, 3, and 4; 3-11, 3-5, 6-8, and 9-11
    types = ['3-11', '3-5', '6-8', '9-11']

    paths = {}
    for t in types:
        paths[t] = glob.glob(path + f'[1-3]/{t}/*.ref')

    data = {}
    for t in types:
        for p in paths[t]:
            sample = Sample()
            with open(p, 'r') as f:
                para = []
                for line in f.readlines():
                    line = line.strip()
                    if line == '==========':
                        if para:
                            sample.add_segment(para)
                        para = []
                    else:
                        para.append(line)
            if t in data:
                data[t].append(sample)
            else:
                data[t] = [sample]
    return data


if __name__ == '__main__':
    # download necessary nltk packages
    # venv_dir = os.getcwd() + '/venv/nltk_data/'
    # nltk_data = ['stopwords', 'punkt']
    # nltk.download(nltk_data, download_dir = venv_dir)

    data_path = '../C99/data/'
    samples = read_samples(data_path)
    tt = TextTiler(20, 6)
    for t in samples:
        for s in samples[t]:
            scores = tt.eval_tile_text(s)
            print(scores)
            exit()
