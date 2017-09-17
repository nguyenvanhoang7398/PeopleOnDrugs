import csv, string
from random import shuffle

from constants import author_doc_review_default_path, thread_doc2vec_model_default_path, author_doc_review_small_1k_default_path,\
    thread_headers, thread_review_csv_default_path, author_doc_review_small_default_path, d2v_dbow_words, d2v_dm_concat, \
    d2v_dm_mean, d2v_iter, d2v_min_count, d2v_negative, d2v_sample, d2v_size, d2v_trim_rule, d2v_window, d2v_workers
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

        thread_review_csvout_writer.writerow(thread_headers)
        for doc_id, thread in thread_dict.items():
            thread_review_csvout_writer.writerow([doc_id, thread])

        csvin.close()
        _thread_review_csvout.close()


class TaggedLineSentence(object):
    def __init__(self, fin):
        self.fin = fin
        self.sentences = []

    def to_array(self):
        csvin = open(self.fin, mode='rt', encoding="ISO-8859-1")  # open csv in
        csvin_reader = csv.reader(csvin, delimiter=',')

        for i, row in enumerate(csvin_reader):
            if i % 50000 == 0:
                print("Processed " + str(i) + " rows")
            if i == 0:
                continue
            doc_id = row[0]
            thread = row[1]
            self.sentences.append(TaggedDocument([word.lower() for word in word_tokenize(thread)
                                                  if word not in string.punctuation and len(word) > 1], [doc_id]))

        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def create_doc2vec_model(fin, d2v_path):
    if isfile(d2v_path):
        print("Thread doc2vec model found at " + d2v_path)
        return Doc2Vec.load(d2v_path)
    else:
        print("Creating new doc2vec model at " + d2v_path)
        sentences = TaggedLineSentence(fin)
        model = Doc2Vec(min_count=d2v_min_count, window=d2v_window, size=d2v_size, sample=d2v_sample,
                        negative=d2v_negative, workers=d2v_workers, iter=d2v_iter, dm_mean=d2v_dm_mean,
                        dm_concat=d2v_dm_concat, dbow_words=d2v_dbow_words, trim_rule=d2v_trim_rule)
        model.build_vocab(sentences.to_array())
        print("Finish building vocabulary")
        model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)
        print("Finish training doc2vec model")
        model.save(d2v_path)
        print("Trained model saved at " + d2v_path)
        return model


if __name__ == "__main__":
    get_thread_review(fin=author_doc_review_default_path, thread_review_csvout=thread_review_csv_default_path)
    d2v_model = create_doc2vec_model(thread_review_csv_default_path, thread_doc2vec_model_default_path)
