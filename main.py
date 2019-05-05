import argparse
import glob
import os
import nltk
import texttiler
import time
from allennlp.commands.elmo import ElmoEmbedder

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

def nltk_init(in_venv):
    # download necessary nltk packages
    nltk_data = ['stopwords', 'punkt']
    if in_venv:
        venv_dir = os.getcwd() + '/venv/nltk_data/'
        nltk.download(nltk_data, download_dir = venv_dir)
    else:
        nltk.download(nltk_data)

def evaluate(samples, tt, f):
    for t in samples:
        s_time = time.time()
        l = len(samples[t])
        f.write(f'{l} samples of type {t}\n')
        count = 0

        # Tuple(pk, wd, bs)
        mini = [1, 1, 1]
        maxi = [0, 0, 0]
        mean = [0, 0, 0]
        for s in samples[t]:
            scores = tt.eval_tile_text(s)
            for i in range(0, len(scores)):
                mini[i] = min(mini[i], scores[i])
                maxi[i] = max(maxi[i], scores[i])
                mean[i] += scores[i]/l
            print(f'Progress: {count}/{l}  ', end='\r')
            count += 1
        met = (time.time() - s_time)/l
        f.write(f'Mean evaluation time: {met:.4f} seconds\n')
        f.write(' '.join([f'{x:.4f}' for x in mini]))
        f.write('\n')
        f.write(' '.join([f'{x:.4f}' for x in maxi]))
        f.write('\n')
        f.write(' '.join([f'{x:.4f}' for x in mean]))
        f.write('\n')

def evaluate_tt(samples, w, k):
    tt = texttiler.TextTiler(w, k)
    with open('tt.out', 'w') as f:
        f.write(f'Window size: {w}\nBlock size: {k}\n') 
        evaluate(samples, tt, f)

def evaluate_ett(samples, w, k, use_gpu):
    w = 20
    k = 3
    gpu = 0 if use_gpu else -1
    elmo = ElmoEmbedder(cuda_device = gpu)
    ett = texttiler.ELMoTextTiler(w, k, elmo)
    with open('ett.out', 'w') as f:
        f.write(f'Window size: {w}\nBlock size: {k}\n')
        evaluate(samples, ett, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("w", help="window size", type=int)
    parser.add_argument("k", help="block size", type=int)
    parser.add_argument("--use_gpu", help="use gpu", action="store_true")
    parser.add_argument("--in_venv", help="using a venv", action="store_true")
    args = parser.parse_args()

    nltk_init(args.in_venv)
    data_path = '../C99/data/'
    samples = read_samples(data_path)

    evaluate_tt(samples, args.w, args.k)
    evaluate_ett(samples, args.w, args.k, args.use_gpu)
