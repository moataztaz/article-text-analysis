# imports
import os
from text_preprocessing.named_entity import step_ner

# paths config
_project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_data_path = os.path.join(_project_path, "data")

# pipelines
def preprocessing_pipeline(data_dir, output_dir):
    step_ner(data_dir=data_dir,
             output_dir=output_dir)


# search
def search_app(**kwargs):
    print("I am not ready !!")
    pass


# main
if __name__ == "__main__":
    # MODE PREPROCESS OR SEARCH
    MODE = "PREPROCESS"
    # CONFIG
    DATA_DIR = _data_path

    if MODE == "PREPROCESS":
        preprocessing_pipeline(data_dir=DATA_DIR)
    elif MODE == "SEARCH":
        search_app()
