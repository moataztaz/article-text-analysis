import spacy
import json
import random
import pandas
from spacy.language import Language
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, Token, Doc
import os

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(project_path, "data")

with open(os.path.join(data_path, "articles1.csv"), encoding="utf8") as f:
    NEWS1 = pandas.read_csv(f)
    # TODO: replace below with regex detection
    NEWS1.drop(columns=['Unnamed: 0', ], inplace=True)

nlp = spacy.load("en_core_web_sm")

content = NEWS1["content"][0]

print(content)
print(nlp.pipe_names)

with nlp.disable_pipes("tagger", "parser", "lemmatizer"):
    doc = nlp(content)

# nlp.pipe(NEWS1["content"])
print(doc.ents)
entities = []
labels = []

for entity in doc.ents:
    entities.append(entity.text)
    labels.append(entity.label_)

data = {'entity': entities, 'label': labels}
df = pandas.DataFrame(data)

# for doc, context in nlp.pipe(DATA, as_tuples=True):

def article_ner(id):
    list_contents = NEWS1["content"][NEWS1["id"] == id].values
    assert len(list_contents) == 1
    content = list_contents[0]
    with nlp.disable_pipes("tagger", "parser", "lemmatizer"):
        doc = nlp(content)

    entities = []
    labels = []

    for entity in doc.ents:
        entities.append(entity.text)
        labels.append(entity.label_)

    data = {'entity': entities, 'label': labels}
    df = pandas.DataFrame(data)

    return df


if __name__ == '__main__':
    article_index = int(input('Write Article Index :'))
    print(article_ner(id=article_index))

