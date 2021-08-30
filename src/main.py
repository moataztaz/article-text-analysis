# imports
import os
from text_preprocessing.named_entity import step_ner
from decorators import timer
from text_preprocessing.embedding import step_embedding

# paths config
_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_data_path = os.path.join(_project_path, "data")


# pipelines

@timer
def preprocessing_pipeline(data_dir, output_dir, input_text, limit=None):
    step_ner(data_dir=data_dir,
             output_dir=output_dir,
             limit=limit)
    step_embedding(data_dir=data_dir,
             output_dir=output_dir,
             input_text=input_text,
             limit=limit)

# search
def search_app(**kwargs):
    print("I am not ready !!")
    pass


# main
if __name__ == "__main__":
    # MODE PREPROCESS OR SEARCH
    MODE = "PREPROCESS"
    LIMIT = int(os.getenv('LIMIT', 10))
    # CONFIG
    DATA_DIR = _data_path
    input_text = input('Write similarity check text :')

    if MODE == "PREPROCESS":
        preprocessing_pipeline(data_dir=DATA_DIR, output_dir=DATA_DIR, input_text=input_text, limit=1000)
    elif MODE == "SEARCH":
        search_app()
