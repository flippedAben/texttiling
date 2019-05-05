import bisect
import collections
import math
import nltk
import segeval
import string
import tensorflow as tf
import numpy as np

class TextTiler(object):
    '''
    TextTiling algorithm with modifications (not including ELMo).
    '''
    def __init__(self, w, k):
        self.w = w
        self.k = k

    def eval_tile_text(self, sample):
        '''
        Returns a tuple of metric scores (Pk, WinDiff, B).
        '''
        ### Record paragraph break points
        sent_bounds, normed_text = self.get_sb_nt(sample)

        ### Break up text into Pseudosentences
        # this list maps pseudosentence index to beginning token index
        ps_bounds = list(range(0, len(normed_text), self.w))
        pseudosents = [normed_text[i:i + self.w] for i in ps_bounds]

        # discard pseudosents of length < self.w
        if len(pseudosents[-1]) < self.w:
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
            sims.append(self.sim(block_a, block_b))
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

        pk = float(segeval.pk(pred, gold))
        wd = float(segeval.window_diff(pred, gold))
        bs = float(segeval.boundary_similarity(pred, gold, one_minus=True))

        return (pk, wd, bs)

    def get_sb_nt(self, sample):
        # this list maps sentence index to beginning token index
        sent_bounds = [0]
        # list of tokens
        normed_text = []
        for seg in sample.get_segments():
            for sent in seg:
                clean_sent = self.clean_text(sent)
                stemmed_sent = self.stem_text(clean_sent)
                normed_sent = self.discard_stopwords(stemmed_sent)
                sent_bounds.append(len(normed_sent) + sent_bounds[-1])
                normed_text += normed_sent
        del sent_bounds[-1]
        return sent_bounds, normed_text

    def clean_text(self, sent):
        tokens = nltk.tokenize.word_tokenize(sent)
        tokens = [t.lower() for t in tokens]
        no_punct = str.maketrans('', '', string.punctuation)
        tokens = [t.translate(no_punct) for t in tokens]
        tokens = [t for t in tokens if t.isalpha()]
        return tokens

    def stem_text(self, tokens):
        ps = nltk.stem.porter.PorterStemmer()
        stems = [ps.stem(w) for w in tokens]
        return stems

    def discard_stopwords(self, tokens):
        sw = nltk.corpus.stopwords.words('english')
        sw = set(sw)
        result = [w for w in tokens if w not in sw]
        return result

    def sim(self, block_a, block_b):
        a = [token for ps in block_a for token in ps]
        b = [token for ps in block_b for token in ps]
        bow_a = collections.Counter(a)
        bow_b = collections.Counter(b)

        union = bow_a | bow_b
        prod = [bow_a[tok] * bow_b[tok] for tok in union]
        numerator = sum(prod)

        sqsum_a = sum([bow_a[tok]**2 for tok in a])
        sqsum_b = sum([bow_b[tok]**2 for tok in b])
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

class ELMoTextTiler(TextTiler):
    '''
    TextTiling algorithm with modifications and ELMo embeddings
    '''
    def __init__(self, w, k, elmo, sess):
        super().__init__(w,k)
        self.elmo = elmo
        self.sess = sess

    def get_sb_nt(self, sample):
        sent_bounds = [0]
        normed_text = []
        for seg in sample.get_segments():
            # List[List[word_tokens]]
            clean_seg = []
            for sent in seg:
                clean_sent = self.clean_text(sent)
                clean_seg.append(clean_sent)
                normed_sent = self.discard_stopwords(clean_sent)
                sent_bounds.append(len(normed_sent) + sent_bounds[-1])
            normed_text += self.get_elmo_embs(clean_seg)
            print('seg done: ')
        del sent_bounds[-1]
        return sent_bounds, normed_text

    def get_elmo_embs(self, seg_tok):
        # length in number of tokens
        sent_lens = [len(sent_tok) for sent_tok in seg_tok]
        max_sent_len = max(sent_lens)
        in_toks = []
        for sent_tok in seg_tok:
            in_toks.append(sent_tok + [''] * (max_sent_len - len(sent_tok)))

        embs = self.elmo(inputs = {
            'tokens': in_toks,
            'sequence_len': sent_lens
            }, signature = 'tokens', as_dict=True)['elmo']

        # List[List[embeddimgs]]
        seg_emb = self.sess.run(embs)

        # throw away embeddings of stopwords
        sw = nltk.corpus.stopwords.words('english')
        sw = set(sw)
        tokens = []
        for si in range(0, len(in_toks)):
            for ti in range(0, sent_lens[si]):
                if in_toks[si][ti] not in sw:
                    tokens.append(seg_emb[si][ti])

        # a "token" in the ELMo sense is a big vector
        return tokens

    def sim(self, block_a, block_b):
        a = np.array([emb for s in block_a for emb in s])
        b = np.array([emb for s in block_b for emb in s])

        # For ELMo, we take the average of both blocks and compute similarity
        avg_a = self.sess.run(tf.reduce_mean(a, 0))
        avg_b = self.sess.run(tf.reduce_mean(b, 0))
        s = tf.losses.cosine_distance(tf.nn.l2_normalize(avg_a, 0),
                tf.nn.l2_normalize(avg_b, 0), axis=0)
        cos_sim = 1 - self.sess.run(s)
        return cos_sim
