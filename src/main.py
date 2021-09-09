# imports
import os
from text_preprocessing.named_entity import step_ner
from decorators import timer
from text_preprocessing.embedding import step_embedding
from search.Search import step_search


# paths config
_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_data_path = os.path.join(_project_path, "data")


# pipelines

@timer
def preprocessing_pipeline(data_dir, output_dir, limit=None):
    step_ner(data_dir=data_dir,
             output_dir=output_dir,
             limit=limit)
    step_embedding(data_dir=data_dir,
                   output_dir=output_dir,
                   limit=limit)


# search

@timer
def search_app(data_dir, output_dir, text=" "):
    step_search(data_dir=data_dir,
                output_dir=output_dir,
                input_text=text)


# main
if __name__ == "__main__":
    # MODE PREPROCESS OR SEARCH
    MODE = " "
    while MODE.upper() not in ["PREPROCESS", "SEARCH"]:
        MODE = input("Choose mode('PREPROCESS'/'SEARCH') :")
    LIMIT = int(os.getenv('LIMIT', 10))
    # CONFIG
    DATA_DIR = _data_path

    if MODE.upper() == "PREPROCESS":
        preprocessing_pipeline(data_dir=DATA_DIR, output_dir=DATA_DIR, limit=LIMIT)
    elif MODE.upper() == "SEARCH":
        input_text = input('Write similarity check text :')
        search_app(data_dir=DATA_DIR, output_dir=DATA_DIR, text=input_text)
