# imports
import os
from text_preprocessing.named_entity import step_ner
from decorators import timer

# paths config
_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_data_path = os.path.join(_project_path, "data")


# pipelines

@timer
def preprocessing_pipeline(data_dir, output_dir, limit=None):
    step_ner(data_dir=data_dir,
             output_dir=output_dir,
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

    if MODE == "PREPROCESS":
        preprocessing_pipeline(data_dir=DATA_DIR, output_dir=DATA_DIR, limit=LIMIT)
    elif MODE == "SEARCH":
        search_app()
