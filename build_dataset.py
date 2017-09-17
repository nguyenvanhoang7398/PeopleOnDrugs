from utils import create_stylistic_feature_vector, create_affective_feature_vector, save_data, load_data, \
    create_user_feature_vector
from constants import feature_list, stylistic_csvout_default_path, stylistic_fout_default_path, \
    affective_csvout_default_path, affective_fout_default_path, author_doc_review_small_default_path, \
    author_doc_review_default_path, affective_headers, stylistic_headers, user_headers, author_details_default_path, \
    user_csvout_default_path, user_fout_default_path

import os, csv, uuid
import numpy as np


def build_review_dataset(fin, stylistic_csvout, stylistic_fout, affective_csvout, affective_fout):
    csvin = open(fin, mode='rt', encoding="ISO-8859-1") # open csv in
    csvin_reader = csv.reader(csvin, delimiter=',')

    _stylistic_csvout = open(stylistic_csvout, "wt") # open csv out for stylistic and affective
    stylistic_csvout_writer = csv.writer(_stylistic_csvout, delimiter=',', lineterminator='\n')
    _affective_csvout = open(affective_csvout, "wt")
    affective_csvout_writer = csv.writer(_affective_csvout, delimiter=',', lineterminator='\n')

    stylistic_feature_dataset = []
    affective_feature_dataset = []

    for i, row in enumerate(csvin_reader):
        if i % 50000 == 0:
            print("Processed " + str(i) + " rows")
        if i == 0:
            stylistic_row = stylistic_headers # if it's the first line, written row = header
            affective_row = affective_headers
        else:
            author_id = row[0] # parse columns
            doc_id = row[1]
            post = row[2]
            post_id = str(uuid.uuid4())
            stylistic_feature_vector = create_stylistic_feature_vector(features_list=feature_list, doc=post)
            affective_feature_vector = create_affective_feature_vector(doc=post)

            stylistic_row = [author_id, doc_id, post_id, str(list(stylistic_feature_vector))]  # arrange data into row format
            affective_row = [author_id, doc_id, post_id, str(list(affective_feature_vector))]
            stylistic_data_point = [author_id, doc_id, post_id, stylistic_feature_vector] # arrange data into row format
            affective_data_point = [author_id, doc_id, post_id, affective_feature_vector]
            stylistic_feature_dataset.append(stylistic_data_point) # append to current dataset
            affective_feature_dataset.append(affective_data_point)

        stylistic_csvout_writer.writerow(stylistic_row) # write row to csv out
        affective_csvout_writer.writerow(affective_row)

    save_data(path=stylistic_fout, py_object=stylistic_feature_dataset) # save dataset to pickle file
    save_data(path=affective_fout, py_object=affective_feature_dataset)
    csvin.close()
    _stylistic_csvout.close()
    _affective_csvout.close()


def build_user_dataset(fin, user_csvout, user_fout):
    csvin = open(fin, mode='rt', encoding="ISO-8859-1")  # open csv in
    csvin_reader = csv.reader(csvin, delimiter=',')

    _user_csvout = open(user_csvout, "wt")  # open csv out for stylistic and affective
    user_csvout_writer = csv.writer(_user_csvout, delimiter=',', lineterminator='\n')

    user_feature_dataset = []
    for i, row in enumerate(csvin_reader):
        if i % 50000 == 0:
            print("Processed " + str(i) + " rows")
        if i == 0:
            user_row = user_headers
        else:
            user_feature_vector = create_user_feature_vector(row)
            user_row = row + [str(list(user_feature_vector))]
            user_data_point = row + [user_feature_vector]
            user_feature_dataset.append(user_data_point)

        user_csvout_writer.writerow(user_row)

    save_data(path=user_fout, py_object=user_feature_dataset)
    csvin.close()
    _user_csvout.close()


if __name__ == "__main__":
    build_review_dataset(fin=author_doc_review_default_path, stylistic_csvout=stylistic_csvout_default_path,
                         stylistic_fout=stylistic_fout_default_path, affective_csvout=affective_csvout_default_path,
                         affective_fout=affective_fout_default_path)
    build_user_dataset(fin=author_details_default_path, user_csvout=user_csvout_default_path,
                       user_fout=user_fout_default_path)