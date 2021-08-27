import pandas as pd
import spacy
import os


_project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_data_path = os.path.join(_project_path, "data")

nlp = spacy.load("en_core_web_sm")

def step_ner(data_dir, output_dir, output_format='csv'):
    """
    Ap preprocessing step, compute ner for each article in the csv files, reads all csv files in directory
    :param data_dir: input csv files directory
    :param output_dir: output directory
    :param output_format: only csv format is supported
    :return: exit code
    """
    csv_files = os.listdir(data_dir)
    frames = []

    for file in csv_files:
        with open(os.path.join(data_dir, file), encoding="utf8") as f:
            file_news = pd.read_csv(f)
        frames.append(file_news)

    NEWS = pd.concat(frames)
    NEWS_chunk = NEWS.head(3)
    data = []

    def extract_data(row):
        ner_row = ner(row["id"], NEWS_chunk)
        for index, entity_row in ner_row.iterrows():
            data.append([int(row["id"]), entity_row["entity"], entity_row["label"]])

    NEWS_chunk.apply(lambda x: extract_data(x), axis=1)

    step_ner_df = pd.DataFrame(data, columns=['article_id', 'entity', 'entity_label'])
    return step_ner_df


def ner(id,news_dataframe):
    list_contents = news_dataframe["content"][news_dataframe["id"] == id].values
    assert len(list_contents) == 1
    content = list_contents[0]
    doc = nlp(content)
    entities = []
    labels = []

    for entity in doc.ents:
        entities.append(entity.text)
        labels.append(entity.label_)

    data = {'entity': entities, 'label': labels}
    df = pd.DataFrame(data)
    # faster if it return a dictionnary !!
    return df


print(step_ner(_data_path, 0, 0))








