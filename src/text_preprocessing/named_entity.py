import re

from spacy.language import Language
from spacy.tokens import Doc
import pandas as pd
import spacy
import os

_dataset_file_regex = r'articles\d*\.csv'

nlp = spacy.load("en_core_web_sm")


def step_ner(data_dir, output_dir, limit=None):
    """
    A preprocessing method that employs a small language model to get a dataset of entities and their labels for each
    article (distinguished by ID)
    :param data_dir: the directory where the articles data exists
    :param output_dir: the directory where out output is saved
    :param limit: the number of entries(articles) to use for the method
    :return: dataset with attributes ["article Id", "entity", "entity label"]
    """

    csv_files = [f for f in os.listdir(data_dir) if re.search(_dataset_file_regex, f)]
    frames = []

    for file in csv_files:
        with open(os.path.join(data_dir, file), encoding="utf8") as f:
            file_news = pd.read_csv(f)
        frames.append(file_news)

    NEWS = pd.concat(frames)
    # if there is no limit the value is None, however NEWS variable is filtered
    if limit:
        NEWS = NEWS.head(limit)

    news_filtered = NEWS.filter(['content', 'id'], axis=1)
    news_tuples = news_filtered.to_records(index=False)
    data = []

    with nlp.disable_pipes("tagger", "parser", "lemmatizer"):
        data.extend([
            [int(context), entity.text, entity.label_]
            for doc, context in nlp.pipe(news_tuples, as_tuples=True)
            for entity in doc.ents])

    step_ner_df = pd.DataFrame(data, columns=['article_id', 'entity', 'entity_label'])
    step_ner_df.to_csv(r'{}\ner_dataframe.csv'.format(output_dir), index=False, header=True)

    return step_ner_df


def step_ner_component_test(data_dir, output_dir, output_format='csv', limit=None):

    csv_files = [f for f in os.listdir(data_dir) if re.search(_dataset_file_regex, f)]
    frames = []

    for file in csv_files:
        with open(os.path.join(data_dir, file), encoding="utf8") as f:
            file_news = pd.read_csv(f)
        frames.append(file_news)

    NEWS = pd.concat(frames)
    # if there is no limit the value is None, however NEWS variable is filtered
    if limit:
        NEWS = NEWS.head(limit)

    news_filtered = NEWS.filter(['content', 'id'], axis=1)
    news_tuples = news_filtered.to_records(index=False)
    data = []

    Doc.set_extension("id", default=None)

    @Language.component("data_component")
    def data_component(doc):
        for ent in doc.ents:
            print("hi")
            print(doc._.id, ent.text, ent.label_)
            data.append([doc._.id, ent.text, ent.label_])
        return doc

    nlp.add_pipe("data_component", after="ner")
    print(nlp.pipe_names)
    with nlp.disable_pipes("tagger", "parser", "lemmatizer"):
        for doc, context in nlp.pipe(news_tuples, as_tuples=True):
            doc._.id = int(context)

    step_ner_df = pd.DataFrame(data, columns=['article_id', 'entity', 'entity_label'])
    step_ner_df.to_csv(r'{}\ner_dataframe.csv'.format(output_dir), index=False, header=True)

    return step_ner_df


# Testing area
if __name__ == '__main__':
    _project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    _data_path = os.path.join(_project_path, "data")

    print(step_ner(_data_path, _data_path, limit=3))






