from utils import create_stylistic_feature_vector, create_affective_feature_vector, save_data, load_data
from constants import feature_list, stylistic_csvout_default_path, stylistic_fout_default_path, \
    affective_csvout_default_path, affective_fout_default_path, author_doc_review_small_default_path, \
    author_doc_review_default_path
import os, csv, uuid
import numpy as np


def build_review_dataset(fin, stylistic_csvout, stylistic_fout, affective_csvout, affective_fout):
    stylistic_headers = ["author-id", "doc-id", "post-id", "stylistic-feature"]
    affective_headers = ["author-id", "doc-id", "post-id", "affective-feature"]

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


def build_user_dataset(review_dataset_path, users_details_path, out_path, samples_per_files=3000):
    user_details_files = [f for f in os.listdir(users_details_path) if f.endswith('.tsv')]
    review_dataset_files = [f for f in os.listdir(review_dataset_path) if f.endswith('.pickle')]
    review_dataset = []
    user_features_dict = {}
    for i, file in enumerate(review_dataset_files):
        if (i+1) % 20:
            print("Loaded", str(i+1), "/", str(len(review_dataset_files)))
        dataset = load_data(review_dataset_path + file)
        review_dataset += dataset

    for review in review_dataset:
        user_id = review[0]
        review_vector = review[2]
        if user_id not in user_features_dict.keys():
            if not np.isnan(review_vector).any():
                user_features_dict[user_id] = review_vector
        elif not np.isnan(review_vector).any():
            user_features_dict[user_id] = np.add(user_features_dict[user_id], review_vector)
    user_dataset = []
    
    i = 0
    for fin in user_details_files:
        csvin = open(users_details_path + fin, mode='rt', encoding="ISO-8859-1")
        csvin_reader = csv.reader(csvin, delimiter='\t')

        for row in csvin_reader:
            if i % 500 == 0:
                print("Proceeded", i, "/ 15000 users")
            i += 1
            user_id = row[0]
            if user_id not in user_features_dict.keys():
                continue    
            num_reviews = int(row[3])
            num_likes = (row[7])
            if num_likes == 'null':
                num_likes = 0
            else:
                num_likes = int(row[7])
            normalized_user_vector = np.divide(user_features_dict[user_id], num_reviews)
            normalized_likes = num_likes / num_reviews   
            if not (np.isnan(normalized_user_vector).any() or np.isnan(normalized_likes)):
                user_dataset.append([normalized_user_vector, normalized_likes])
        csvin.close()
    small_batch = []
    j = 1
    for i, data in enumerate(user_dataset):
        small_batch.append(data)
        if (i+1) % samples_per_files == 0:
            save_data(out_path + "user_dataset_" + str(j) + ".pickle", small_batch)
            j += 1
            small_batch = []
    save_data(out_path + "user_dataset_" + str(j) + ".pickle", small_batch)


if __name__ == "__main__":
    build_review_dataset(fin=author_doc_review_small_default_path, stylistic_csvout=stylistic_csvout_default_path,
                         stylistic_fout=stylistic_fout_default_path, affective_csvout=affective_csvout_default_path,
                         affective_fout=affective_fout_default_path)