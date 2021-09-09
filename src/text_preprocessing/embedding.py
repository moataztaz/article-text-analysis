import pandas as pd
import spacy
import os
import re
import json
import numpy as np


_dataset_file_regex = r'articles\d*\.csv'

nlp = spacy.load("en_core_web_md")


def step_embedding(data_dir, output_dir, limit=None):
    """
    A preprocessing method that employs a larger language model to get a list of articles(distinguished by ID) and their
    corresponding vector.
    :param data_dir: the directory where the articles data exists
    :param output_dir: the directory where our output is saved
    :param limit: the number of entries(articles) to use for the method
    :return: list of attributes ["article Id", "article vector"]
    """

    csv_files = [f for f in os.listdir(data_dir) if re.search(_dataset_file_regex, f)]
    frames = []

    for file in csv_files:
        with open(os.path.join(data_dir, file), encoding="utf8") as f:
            file_news = pd.read_csv(f)
        frames.append(file_news)

    NEWS = pd.concat(frames)
    if limit:
        NEWS = NEWS.head(limit)

    news_tuples = NEWS.filter(['content', 'id'], axis=1).to_records(index=False)
    data = []

    data.extend([
        [int(context), np.ndarray.tolist(doc.vector)]
        for doc, context in nlp.pipe(news_tuples, as_tuples=True)])

    with open(os.path.join(output_dir, "embedding_dataframe.json"), "w", encoding="utf8") as json_file:
        json.dump(data, json_file, sort_keys=True)

    return data


def similarity_by_id_test(id, input_text, news_dataframe):
    list_contents = news_dataframe["content"][news_dataframe["id"] == id].values
    assert len(list_contents) == 1
    content = list_contents[0]
    doc = nlp(content)
    text = nlp(input_text)
    similarity = text.similarity(doc)

    return similarity


# Testing Area:

if __name__ == '__main__':
    _project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    _data_path = os.path.join(_project_path, "data")

    print(step_embedding(_data_path, _data_path, limit=3))