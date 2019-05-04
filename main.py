import texttiler
import glob
import os

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

if __name__ == '__main__':
    # download necessary nltk packages
    # venv_dir = os.getcwd() + '/venv/nltk_data/'
    # nltk_data = ['stopwords', 'punkt']
    # nltk.download(nltk_data, download_dir = venv_dir)

    data_path = '../C99/data/'
    samples = read_samples(data_path)
    tt = texttiler.TextTiler(20, 6)
    for t in samples:
        for s in samples[t]:
            scores = tt.eval_tile_text(s)
            print(scores)
            exit()
