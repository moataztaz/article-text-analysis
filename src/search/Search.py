# imports
import json
import os
from operator import itemgetter
import pandas as pd
import spacy
import numpy as np
from scipy import spatial

# paths config
_project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_data_path = os.path.join(_project_path, "data")

nlp = spacy.load("en_core_web_md")


def step_search(data_dir, output_dir, input_text):
    """
    A search function that uses preprocessed data to get a sorted list of article IDs depending on their cosine
    similarity to the input text (sorted from most to least similar)
    :param data_dir: the directory where the articles data exists
    :param output_dir: the directory where our output is saved
    :param input_text: the text that will be compared to the preprocessed data
    :return: dataset with attributes: ["article Id", "article similarity"]
    """

    input_vector = nlp(input_text).vector

    # read embeddings file
    with open(os.path.join(data_dir, "embedding_dataframe.json"), encoding="utf8") as f:
        news_embedding = json.load(f)

    similarity_list = []
    similarity_list.extend(
        [[id, vector_similarity(input_vector, np.asarray(embedding, dtype="float32"))] for id, embedding in
         news_embedding])

    sorted_list = sorted(similarity_list, key=itemgetter(1), reverse=True)
    search_df = pd.DataFrame(sorted_list, columns=['article_id', 'similarity'])
    search_df.to_csv(r'{}\input_text_similarity.csv'.format(output_dir), index=False, header=True)
    # search_df.to_csv(r'{}\{}_similarity.csv'.format(output_dir, input_text.replace(" ", "_")), index=False,
    # header=True)

    return search_df


# Function that calculates cosine similarity between two vectors
def vector_similarity(input_vector, vector_dict):
    return 1 - spatial.distance.cosine(input_vector, vector_dict)


# Testing Area:
if __name__ == '__main__':
    print(step_search(_data_path, _data_path, "hello it's me, Mario"))


