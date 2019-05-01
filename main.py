import glob
import spacy
from spacy.lang.en import English

class Sample(object):
    def __init__(self):
        self.segments = []

    def add_segment(self, p):
        self.segments.append(p)

    def get_first(self):
        return self.segments[0]

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


def tokenize(doc):
    nlp = English()
    tokenizer = English().Defaults.create_tokenizer(nlp)

    print(tokenizer(doc))

if __name__ == '__main__':
    data_path = '../C99/data/'
    docs = read_docs(data_path)
    for t in docs:
        for d in docs[t]:
            tokenize(d.get_first())
            exit()
