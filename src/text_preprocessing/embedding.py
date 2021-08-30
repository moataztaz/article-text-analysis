import pandas as pd
import spacy
import os
import re


_project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_data_path = os.path.join(_project_path, "data")
_dataset_file_regex = r'articles\d*\.csv'

nlp = spacy.load("en_core_web_md")


def step_embedding(data_dir, output_dir, input_text, limit=None):
    """
    A preprocessing method that employs embeddings to get a list of tuples describing the similarity of each news
    article to a given text (article Id, similarity)
    :param data_dir: input csv files directory
    :param input_text: input text to compare similarity with
    :return:
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
    data = []

    def fill_list(row):
        data.append([int(row["id"]), similarity_by_id(row["id"], input_text, NEWS)])

    # similarity_list = NEWS.apply(lambda x: similarity_by_id(x["id"], input_text, NEWS), axis=1)
    NEWS.apply(lambda x: fill_list(x), axis=1)
    similarity_df = pd.DataFrame(data, columns=["article_id", "similarity_score"])
    similarity_df.to_csv(r'{}\embedding_dataframe.csv'.format(output_dir), index=False, header=True)

    return similarity_df


def similarity_by_id(id, input_text, news_dataframe):
    list_contents = news_dataframe["content"][news_dataframe["id"] == id].values
    assert len(list_contents) == 1
    content = list_contents[0]
    doc = nlp(content)
    text = nlp(input_text)
    similarity = text.similarity(doc)

    return similarity


# Testing Area:

