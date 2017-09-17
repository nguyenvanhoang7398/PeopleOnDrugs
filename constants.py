
# default paths
affective_word_dict_default_path = "processed_data/affective_word_dict.pickle"
affective_feature_default_path = "data/Affective-Features.csv"
author_doc_review_default_path = "data/Author-Doc-Review.csv"
author_doc_review_small_default_path = "data/Author-Doc-Review-Small.csv"
author_details_default_path = "data/Author-Details.csv"
author_details_small_default_path = "data/Author-Details-Small.csv"
stylistic_csvout_default_path = "processed_data/Author-Doc-Stylistic.csv"
affective_csvout_default_path = "processed_data/Author-Doc-Affective.csv"
user_csvout_default_path = "processed_data/Author-Feature.csv"
stylistic_fout_default_path = "processed_data/stylistic_feature.pickle"
affective_fout_default_path = "processed_data/affective_feature.pickle"
user_fout_default_path = "processed_data/user_feature.pickle"
user_feature_dict_default_path = "processed_data/user_feature_dict.pickle"
thread_review_csv_default_path = "processed_data/Thread-Review.csv"

# model default paths
thread_doc2vec_model_default_path = "doc2vec_model/thread_doc2vec.d2v"


# csv headers
stylistic_headers = ["author-id", "doc-id", "post-id", "stylistic-feature"]
affective_headers = ["author-id", "doc-id", "post-id", "affective-feature"]
user_headers = ["author-id", "gender", "location", "#posts", "membership-types", "#questions", "#replies", "#thanks", "user-feature"]
thread_headers = ["doc-id", "post"]

# numeric constants
num_user_features = 6
user_max_posts = 12921
user_max_questions = 608
user_max_replies = 8154
user_max_thanks = 2462


feature_list = [
    ["might", "could", "can", "would", "may"],
    ["should", "ought", "need", "shall", "will", "must"],
    ["if"],
    ["no", "not", "neither", "nor", "never"],
    ["therefore", "furthermore", "consequently", "thus", "as", "subsequently", "eventually", "hence"],
    ["till", "until", "despite", "inspite", "though", "although"],
    ["but", "however", "nevertheless", "otherwise", "yet", "still", "nonetheless"],
    ["i", "we", "me", "us", "my", "mine", "our", "ours"],
    ["you, your, yours"],
    ["he", "she", "him", "her", "his", "it", "its", "hers", "they", "them", "their", "theirs"],
    ["DT", "the", "this", "that", "these", "those"],
    ["WDT", "WP", "WP$", "WRB", "?"],
    ["JJ", "JJR", "JJS"],
    ["RB", "RBR", "RBS"],
    ["NNP", "NNPS"]
]


idx_dict = {
    "Strong modals": 0,
    "Weak modals": 1,
    "Conditionals": 2,
    "Negation": 3,
    "Inferential Conjunctions": 4,
    "Contrasting Conjunctions": 5,
    "Following Conjunctions": 6,
    "First Person": 7,
    "Second Person": 8,
    "Third Person": 9,
    "Determiner": 10,
    "QS": 11,
    "Adjectives": 12,
    "Adverbs": 13,
    "Proper Nouns": 14
}

schema = {
    "Author-Drug-Docs.tsv": ["author-id", "drug", "doc-id-list"],
    "Author-Details.tsv": ["author-id", "gender", "location", "#posts", "membership-type", "#questions", "#replies",
                           "#thanks"],
    "Author-Doc-Review.tsv": ["author-id", "doc-id", "post"],
    "Author-Doc-Symtpoms.tsv": ["author-id", "doc-id", "symptoms-sentence-delimited-by-#"],
    "Expert-Drug-SideEffects.tsv": ["drug-family", "SideEffect-Category", "Side-Effect-List",
                                    "UMLS-Concept-Mapping-Each-Item-delimited-by-#"],
    "Author-Xanax-SideEffects.tsv": ["author-id", "doc-id", "symptom-list-with-frequency-in-post"],
    "Author-Ibuprofen-SideEffects.tsv": ["author-id", "doc-id", "symptom-list-with-frequency-in-post"],
    "Author-Prilosec-SideEffects.tsv": ["author-id", "doc-id", "symptom-list-with-frequency-in-post"],
    "Author-Metformin-SideEffects.tsv": ["author-id", "doc-id", "symptom-list-with-frequency-in-post"],
    "Author-Tirosint-SideEffects.tsv": ["author-id", "doc-id", "symptom-list-with-frequency-in-post"],
    "Author-Flagyl-SideEffects.tsv": ["author-id", "doc-id", "symptom-list-with-frequency-in-post"],
    "Affective-Features.tsv": ["affective-category", "affective-word-list"]
}