import collections
import glob
import math
import nltk
import os
import string

# A single sample from the Choi 2000 data
class Sample(object):
    def __init__(self):
        self.segments = []

    def add_segment(self, p):
        self.segments.append(p)

    def get_segments(self):
        return self.segments

    def get_all(self):
        return ' '.join(self.segments)


# TextTiling algorithm with modifications (not including ELMo)
class TextTiler(object):
    def __init__(self, w, k):
        self.w = w
        self.k = k

    def tile_text(self, sample):
        ### Record paragraph break points
        # this list maps segment index to beginning token index
        pbreaks = [0]
        normed_text = []
        for seg in sample.get_segments():
            normed_seg = self.normalize_text(seg)
            normed_text += normed_seg
            pbreaks.append(len(normed_seg) + pbreaks[-1])
        del pbreaks[-1]

        ### Break up text into Pseudosentences
        # this list maps pseudosentence index to beginning token index
        tidx = list(range(0, len(normed_text), self.w))
        pseudosents = [normed_text[i:i + self.w] for i in tidx]

        # discard pseudosents of length < self.w
        # also, record the waste for fun
        waste = 0
        if len(pseudosents[-1]) < self.w:
            waste += len(pseudosents[-1])
            print(f'Waste so far: {waste} tokens')
            del pseudosents[-1]

        ### Group into blocks and calculate sim scores
        sims = []
        i = 0
        while i + 2 * self.k < len(pseudosents):
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
        exit()

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
                            sample.add_segment(' '.join(para))
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
            tiled = tt.tile_text(s)
            print(tiled)
            exit()
