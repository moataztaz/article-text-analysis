import pandas as pd
import spacy
import os


_project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_data_path = os.path.join(_project_path, "data")

nlp = spacy.load("en_core_web_md")


def step_embedding(data_dir, input_text):
    """
    A preprocessing method that employs embeddings to get a list of tuples describing the similarity of each news
    article to a given text (article Id, similarity)
    :param data_dir: input csv files directory
    :param input_text: input text to compare similarity with
    :return:
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

    def fill_list(row):
        data.append([int(row["id"]), similarity_by_id(row["id"], input_text, NEWS_chunk)])

    # similarity_list = NEWS_chunk.apply(lambda x: similarity_by_id(x["id"], input_text, NEWS_chunk), axis=1)
    NEWS_chunk.apply(lambda x: fill_list(x), axis=1)
    similarity_df = pd.DataFrame(data, columns=["article_id", "similarity_score"])

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

with open(os.path.join(_data_path, "articles1.csv"), encoding="utf8") as f:
    NEWS1 = pd.read_csv(f)
    # TODO: replace below with regex detection
    NEWS1.drop(columns=['Unnamed: 0', ], inplace=True)

print(similarity_by_id(17283, "how is life in England", NEWS1))

print("articles similarity to the input is: \n", step_embedding(_data_path, "how is life in England"))
