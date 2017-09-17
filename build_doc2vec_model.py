import csv

from constants import author_doc_review_default_path, thread_doc2vec_model_default_path, \
    thread_headers, thread_review_csv_default_path, author_doc_review_small_default_path
from utils import isfile, word_tokenize

from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


def get_thread_review(fin, thread_review_csvout):
    if isfile(thread_review_csvout):
        print("Thread review found at " + thread_review_csvout)
    else:
        csvin = open(fin, mode='rt', encoding="ISO-8859-1")  # open csv in
        csvin_reader = csv.reader(csvin, delimiter=',')

        _thread_review_csvout = open(thread_review_csvout, "wt")  # open csv out for stylistic and affective
        thread_review_csvout_writer = csv.writer(_thread_review_csvout, delimiter=',', lineterminator='\n')

        thread_dict = {}

        for i, row in enumerate(csvin_reader):
            if i % 50000 == 0:
                print("Processed " + str(i) + " rows")
            if i == 0:
                continue
            doc_id = row[1]
            post = row[2]

            if doc_id not in thread_dict.keys():
                thread_dict[doc_id] = str(post)
            else:
                thread_dict[doc_id] += " " + str(post)

        for doc_id, thread in thread_dict.items():
            thread_review_csvout_writer.writerow([doc_id, thread])

        csvin.close()
        _thread_review_csvout.close()


class TaggedLineSentence(object):
    def __init__(self, fin):
        self.fin = fin


def create_doc2vec_model(fin):
    if isfile(fin):
        print("Thread doc2vec model found at " + fin)
        return Doc2Vec.load(fin)
    else:
        pass


if __name__ == "__main__":
    get_thread_review(fin=author_doc_review_small_default_path, thread_review_csvout=thread_review_csv_default_path)
    # create_doc2vec_model(author_doc_review_default_path)
