# imports
import json
import os
import pandas as pd
import spacy
import numpy as np

# paths config
_project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_data_path = os.path.join(_project_path, "data")

nlp = spacy.load("en_core_web_md")

def step_search(data_dir, output_dir, input_text):

    input_vector = nlp(input_text).vector

    # read entity recognition file
    with open(os.path.join(data_dir, "ner_dataframe.csv"), encoding="utf8") as f:
        news_ner = pd.read_csv(f)

    # read embeddings file
    with open(os.path.join(data_dir, "embedding_dataframe.json"), encoding="utf8") as f:
        news_embedding = json.load(f)

    similarity_list = []
    similarity_list.extend([[id, vector_similarity(input_vector, np.asarray(embedding, dtype="float32"))] for id, embedding in news_embedding])

    return similarity_list.sort(key=similarity_list[1], reverse=True)


def vector_similarity(input_vector, vector_dict):
    similarity = input_vector.similarity(vector_dict)

    return similarity


# Testing Area:

if __name__ == '__main__':
    with open(os.path.join(_data_path, "embedding_dataframe.json"), "r") as f:
        news_embedding = json.load(f)
    for a, b in news_embedding:
        c = np.asarray(b, dtype="float32")
        print(a, " space ", type(c), " space ", type(c[0]))
