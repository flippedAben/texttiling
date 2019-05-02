import glob
import nltk
import string

class Sample(object):
    def __init__(self):
        self.segments = []

    def add_segment(self, p):
        self.segments.append(p)

    def get_segments(self):
        return self.segments

    def get_all(self):
        return ' '.join(self.segments)


def read_docs(path):
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
            doc = Sample()
            with open(p, 'r') as f:
                para = []
                for line in f.readlines():
                    line = line.strip()
                    if line == '==========':
                        if para:
                            doc.add_segment(' '.join(para))
                        para = []
                    else:
                        para.append(line)
            if t in data:
                data[t].append(doc)
            else:
                data[t] = [doc]
    return data


def normalize_text(doc):
    tokens = nltk.tokenize.word_tokenize(doc)
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


if __name__ == '__main__':
    # only need to run the following line once
    # nltk.download()
    data_path = '../C99/data/'
    docs = read_docs(data_path)
    for t in docs:
        for sample in docs[t]:
            clean_paras = []
            for para in sample.get_segments():
                clean_paras.append(normalize_text(para))
            exit()
